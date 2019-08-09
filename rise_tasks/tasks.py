import datetime

import numpy as np
from bson import objectid
from loguru import logger
from pymongo import MongoClient
from pymongo.database import Database

import utils
from app import celery, rs_client, hs_client, MONGO_PORT, MONGO_URL
from client import HydroServingServable
from rise.rise import RiseImageExplainer


@celery.task(bind=True)
def rise_task(self, explanation_id: str):
    mongo_client: MongoClient = MongoClient(host=MONGO_URL, port=MONGO_PORT)
    db: Database = mongo_client['root_cause']

    explanation_id = objectid.ObjectId(explanation_id)
    job_json = db.rise_explanations.find_one_and_update({"_id": explanation_id},
                                                        {"$set": {"started_at": datetime.datetime.now()}})

    if 'result' in job_json:
        return str(explanation_id)

    model_name = job_json['model']['name']
    model_version = job_json['model']['version']
    model = hs_client.get_model(model_name, model_version)

    ts = job_json['explained_instance']['timestamp']
    folder = str(model.id)
    reqstore_uid = job_json['explained_instance']['uid']

    if model.contract.number_of_input_tensors != 1:
        raise ValueError("Unable to explain multi-tensor models")

    if 'probabilities' not in model.contract.output_names:
        raise ValueError("Model have to return probabilities tensor")

    input_tensors = utils.get_reqstore_request_tensors(model.contract, rs_client, folder, ts, reqstore_uid)

    if len(input_tensors) != 1:
        raise ValueError("Request has to many input tensors")

    image: np.ndarray = list(input_tensors.values())[0]

    logger.info(
        f"Initiated task to explain image folder:{folder}, ts:{ts}, uid:{reqstore_uid} under model {model_name}_{model_version} with rise")

    # Create temporary servable
    temp_servable_copy: HydroServingServable = hs_client.deploy_servable(model_name, model_version)

    explained_image_probas = temp_servable_copy(image)['probabilities'][0]  # Reduce batch dim
    top_10_classes = explained_image_probas.argsort()[::-1][:10]
    top_10_probas = explained_image_probas[top_10_classes]

    input_tensor_name = temp_servable_copy.contract.input_names[0]

    input_shape = temp_servable_copy.contract.input_shapes[input_tensor_name]
    input_shape = input_shape[1:]  # Remove 0 dimension as batch dim and

    if len(input_shape) == 2:
        is_single_channel = True
    elif len(input_shape) == 3:
        is_single_channel = False
        input_shape = input_shape[:2]

    else:
        raise ValueError(f"Unable to explain image models with shape {input_shape}")

    rise_config = {"input_size": input_shape,
                   "number_of_masks": 1500,
                   "mask_granularity": 15,
                   "mask_density": 0.4,
                   "single_channel": is_single_channel}

    logger.info(f"Rise config is: {str(rise_config)}")

    rise_explainer = RiseImageExplainer()

    # FIXME add classifier_fn support for contracts without batch dim
    classifier_fn = lambda x: temp_servable_copy(x)['probabilities']

    rise_explainer.fit(prediction_fn=classifier_fn,
                       input_size=rise_config['input_size'],
                       number_of_masks=rise_config['number_of_masks'],
                       mask_granularity=rise_config['mask_granularity'],
                       mask_density=rise_config['mask_density'],
                       single_channel=rise_config['single_channel'])

    def state_updater(x):
        self.update_state(state='STARTED', meta={'progress': x})

    saliency_maps: np.array = rise_explainer.explain(image, state_updater=state_updater)

    # normalize saliency map to (0;1)
    _min = np.min(saliency_maps)
    _max = np.max(saliency_maps)
    saliency_maps = (saliency_maps - _min) / (_max - _min)

    np.rint(saliency_maps * 255, out=saliency_maps)  # Round to int pixel values
    saliency_maps = saliency_maps.astype(np.uint8)  # Use corresponding dtype
    saliency_maps = saliency_maps.reshape((-1, saliency_maps.shape[1] * saliency_maps.shape[2]))  # Flatten masks

    top_10_saliency_maps = saliency_maps[top_10_classes]

    result = [{"mask": m, "class": c, "probability": p} for m, c, p in
              zip(top_10_saliency_maps.tolist(), top_10_classes.tolist(), top_10_probas.tolist())]

    # Store explanation in MongoDB
    db.rise_explanations.update_one({"_id": explanation_id}, {"$set": {'result': result,
                                                                       "completed_at": datetime.datetime.now()}})

    logger.info(f"Finished task to explain {model_name}_{model_version} with rise")

    temp_servable_copy.delete()
    return str(explanation_id)
