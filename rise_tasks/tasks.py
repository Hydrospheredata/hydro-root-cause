import asyncio
import datetime
import logging as logger

import numpy as np
import requests
from bson import objectid
from hydrosdk.modelversion import ModelVersion
from hydrosdk.servable import Servable
from pymongo import MongoClient
from pymongo.database import Database

import utils
from app import celery, hs_cluster, get_mongo_client, MONITORING_URL
from rise.rise import RiseImageExplainer


def get_rise_explained_instance(request_id, feature_name):
    request_with_checks = requests.get(f"{MONITORING_URL}/checks/{request_id}").json()
    return np.array(request_with_checks[feature_name])


@celery.task(bind=True)
def rise_task(self, explanation_id: str):
    mongo_client: MongoClient = get_mongo_client()
    db: Database = mongo_client['root_cause']

    job_json = db.anchor.find_one_and_update({"_id": objectid.ObjectId(explanation_id)},
                                             {"$set": {"started_at": datetime.datetime.now(),
                                                       "status": "STARTED"}})

    model_version_id = int(job_json['model_version_id'])
    request_id = job_json['explained_request_id']
    logger.info(f"{explanation_id} - initiated task to explain image req_id:{request_id} under model version {model_version_id} with rise")

    # Fetch config for this model version. If no config provided - get default one
    job_config = db.configs.find_one({"method": "rise",
                                      "model_version_id": model_version_id})
    if not job_config:
        job_config = utils.get_default_config("rise")

    model_version = ModelVersion.find_by_id(hs_cluster, model_version_id)
    input_field_names = [t.name for t in model_version.contract.predict.inputs]

    # TODO in future pass input parameter to be explained as config?
    input_image: np.ndarray = get_rise_explained_instance(request_id, input_field_names[0])

    # Create temporary servable
    tmp_servable: Servable = Servable.create(hs_cluster, model_name=model_version.name, version=model_version.version,
                                   metadata={"created_by": "rootcause"})
    logger.info(f"{explanation_id} - created servable {tmp_servable.name}")
    utils.servable_lock_till_serving(hs_cluster, tmp_servable.name)
    tmp_servable = Servable.find_by_name(hs_cluster, tmp_servable.name)

    predictor = None  # TODO get_temp_servable_copy_predictor

    explained_image_probas = predictor.predict(input_image)[job_json['explained_output_field_name']][0]  # Reduce batch dim

    # Return only top 10 classes as a result, to reduce response size
    top_10_classes = explained_image_probas.argsort()[::-1][:10]
    top_10_probas = explained_image_probas[top_10_classes]

    input_shape = tuple(map(lambda dim: dim.size, tmp_servable.model.contract.predict.inputs[0].shape.dim))
    input_shape = input_shape[1:]  # Remove 0 dimension as batch dim, FIXME tech debt, support for single data point in future?

    if len(input_shape) == 2:
        is_single_channel = True
    elif len(input_shape) == 3:
        if input_shape[-1] == 1:
            is_single_channel = True
        elif input_shape[-1] == 3:
            is_single_channel = False
        else:
            raise ValueError(f"Invalid number of channels, shape: {input_shape}")
        input_shape = input_shape[:2]
    else:
        raise ValueError(f"Unable to explain image models with shape {input_shape}")

    rise_explainer = RiseImageExplainer()

    def classifier_fn(x):
        return predictor.predict(x)[job_json['explained_output_field_name']]

    rise_explainer.fit(prediction_fn=classifier_fn,
                       input_size=job_config['input_size'],
                       number_of_masks=job_config['number_of_masks'],
                       mask_granularity=job_config['mask_granularity'],
                       mask_density=job_config['mask_density'],
                       single_channel=job_config['single_channel'])

    def state_updater(x):
        self.update_state(state='STARTED', meta={'progress': x})

    saliency_maps: np.array = rise_explainer.explain(input_image, state_updater=state_updater)

    saliency_maps = saliency_maps.reshape((-1, saliency_maps.shape[1] * saliency_maps.shape[2]))  # Flatten masks to return to UI

    top_10_saliency_maps = saliency_maps[top_10_classes]

    result = [{"mask": np.uint8(mask * 255).tolist(),
               "class": c,
               "probability": p} for mask, c, p in
              zip(top_10_saliency_maps, top_10_classes.tolist(), top_10_probas.tolist())]

    db.rise.update_one({"_id": explanation_id}, {"$set": {'result': result,
                                                          "completed_at": datetime.datetime.now()}})

    logger.info(f"{explanation_id} - finished task for {model_version} {request_id}")

    Servable.delete(hs_cluster, tmp_servable.name)

    return str(explanation_id)
