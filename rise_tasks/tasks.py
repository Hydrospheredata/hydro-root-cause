import datetime
from typing import Callable

import numpy as np
from bson import objectid
from hydrosdk.model import Model
from hydrosdk.predictor import AbstractPredictor
from hydrosdk.servable import Servable
from loguru import logger
from pymongo import MongoClient
from pymongo.database import Database

import utils
from app import celery, hs_cluster, get_mongo_client
from rise.rise import RiseImageExplainer


def get_rise_classifier_fn(predictor: AbstractPredictor) -> Callable:
    # TODO fixme predictor
    # TODO add batching support
    # TODO check for batch_dim shape

    return lambda x: predictor.predict(x)['probabilities']


@celery.task(bind=True)
def rise_task(self, explanation_id: str):
    mongo_client: MongoClient = get_mongo_client()
    db: Database = mongo_client['root_cause']

    explanation_id = objectid.ObjectId(explanation_id)
    job_json = db.rise.find_one_and_update({"_id": explanation_id},
                                           {"$set": {"started_at": datetime.datetime.now()}})

    if 'result' in job_json:
        return str(explanation_id)

    job_json = db.anchor.find_one_and_update({"_id": explanation_id},
                                             {"$set": {"started_at": datetime.datetime.now(),
                                                       "status": "STARTED"}})

    model_version_id = job_json['model_version_id']
    request_id = job_json['explained_request_id']

    model_version = Model.find_by_id(hs_cluster, model_version_id)

    # FIXME load training data from s3 instead
    input_tensor = None

    image: np.ndarray = next(input_tensor.values())

    logger.info(f"Initiated task to explain image req_id:{request_id} under model version {model_version} with rise")

    # Create temporary servable
    temp_servable_copy: Servable = Servable.create(hs_cluster, model_version.name, model_version.version)
    # TODO get_temp_servable_copy_predictor ...
    predictor = None

    explained_image_probas = predictor.predict(image)['probabilities'][0]  # Reduce batch dim

    # Return only top 10 classes as a result, to reduce response size
    top_10_classes = explained_image_probas.argsort()[::-1][:10]
    top_10_probas = explained_image_probas[top_10_classes]

    input_shape = tuple(map(lambda dim: dim.size, temp_servable_copy.model.contract.predict.inputs[0].shape.dim))
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

    # Fetch config for this model version. If no config provided - get default one
    job_config = db.configs.find_one({"method": "rise",
                                      "model_version_id": model_version_id})
    if not job_config:
        job_config = utils.get_default_config("rise")

    logger.info(f"Rise config is: {str(job_config)}")

    rise_explainer = RiseImageExplainer()

    classifier_fn = get_rise_classifier_fn(predictor)

    rise_explainer.fit(prediction_fn=classifier_fn,
                       input_size=job_config['input_size'],
                       number_of_masks=job_config['number_of_masks'],
                       mask_granularity=job_config['mask_granularity'],
                       mask_density=job_config['mask_density'],
                       single_channel=job_config['single_channel'])

    def state_updater(x):
        self.update_state(state='STARTED', meta={'progress': x})

    saliency_maps: np.array = rise_explainer.explain(image, state_updater=state_updater)

    # def normalize(x):
    # Normalize masks (?)
    #     _min = np.min(x)
    #     _max = np.max(x)
    #     x = (x - _min) / (_max - _min)
    #     return x

    saliency_maps = saliency_maps.reshape((-1, saliency_maps.shape[1] * saliency_maps.shape[2]))  # Flatten masks to return to UI

    top_10_saliency_maps = saliency_maps[top_10_classes]

    result = [{"mask": np.uint8(mask * 255).tolist(),
               "class": c,
               "probability": p} for mask, c, p in
              zip(top_10_saliency_maps, top_10_classes.tolist(), top_10_probas.tolist())]

    # Store explanation in MongoDB
    db.rise.update_one({"_id": explanation_id}, {"$set": {'result': result,
                                                          "completed_at": datetime.datetime.now()}})

    logger.info(f"{model_version} finished explanation task ")

    Servable.delete(hs_cluster, temp_servable_copy.name)

    return str(explanation_id)
