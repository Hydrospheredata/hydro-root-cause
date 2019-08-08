import datetime

import numpy as np
from bson import objectid
from loguru import logger
from pymongo import MongoClient
from pymongo.database import Database

from app import celery, rs_client, hs_client, MONGO_PORT, MONGO_URL
from client import HydroServingServable
from rise.rise import RiseImageExplainer
import utils


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

    input_tensor_name = temp_servable_copy.contract.input_names[0]

    input_dims = temp_servable_copy.contract.input_shapes[input_tensor_name]
    input_dims = input_dims[1:]  # Remove 0 dimension as batch dim

    if len(input_dims) == 2:
        is_single_channel = True
    elif len(input_dims) == 3:
        is_single_channel = False
    else:
        raise ValueError(f"Unable to explain image models with shape {input_dims}")

    rise_config = {"input_size": input_dims,
                   "number_of_masks": 1000,
                   "mask_granularity": 10,
                   "mask_density": 0.5,
                   "single_channel": is_single_channel}

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

    saliency_map: np.array = rise_explainer.explain(image, state_updater=state_updater)

    # Since we do not need precise saliency maps, we can round them
    np.round(saliency_map, 3, out=saliency_map)

    # normalize saliency map to (0;1)
    _min = np.min(saliency_map)
    _max = np.max(saliency_map)
    saliency_map = (saliency_map - _min) / (_max - _min)

    np.rint(saliency_map * 255, out=saliency_map)  # Round to int pixel values
    saliency_map = saliency_map.astype(np.uint8)  # Use corresponding dtype

    result_json = {"masks": saliency_map.tolist()}

    # Store explanation in MongoDB
    db.rise_explanations.update_one({"_id": explanation_id}, {"$set": {'result': result_json,
                                                                       "completed_at": datetime.datetime.now()}})

    logger.info(f"Finished task to explain {model_name}_{model_version} with rise")

    temp_servable_copy.delete()
    return str(explanation_id)
