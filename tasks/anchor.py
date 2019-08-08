import datetime
from typing import Dict, List

import numpy as np
from anchor2 import TabularExplainer
from bson import objectid
from loguru import logger
from pymongo import MongoClient
from pymongo.database import Database

import utils
from app import celery, rs_client, hs_client, MONGO_PORT, MONGO_URL
from client import HydroServingServable


@celery.task(bind=True)
def anchor_task(self, explanation_id: str):
    #  TODO if mongo anchor cache is available we can check anchor presence by binary mask
    #  ex: db.collection.save({ _id: 1, a: 54, binaryValueofA: "00110110" })
    #  ex: db.collection.find( { a: { $bitsAllSet: [ 1, 5 ] } } )

    mongo_client: MongoClient = MongoClient(host=MONGO_URL, port=MONGO_PORT)
    db: Database = mongo_client['root_cause']

    explanation_id = objectid.ObjectId(explanation_id)
    job_json = db.anchor_explanations.find_one_and_update({"_id": explanation_id},
                                                          {"$set": {"started_at": datetime.datetime.now()}})

    if 'result' in job_json:
        return str(explanation_id)

    model_name = job_json['model']['name']
    model_version = job_json['model']['version']
    config = job_json.get('config', {})

    model = hs_client.get_model(model_name, model_version)

    ts = job_json['explained_instance']['timestamp']
    folder = str(model.id)
    reqstore_uid = job_json['explained_instance']['uid']

    logger.info(
        f"Initiated task to explain numerical sample folder:{folder}, ts:{ts},"
        f" uid:{reqstore_uid} under model {model_name}_{model_version} with anchor")

    if 'classes' not in model.contract.output_names:
        raise ValueError("Model have to return 'classes' tensor of shape [-1; 1]")

    input_tensors = utils.get_reqstore_request_tensors(model.contract, rs_client, folder, ts, reqstore_uid)

    input_array = input_tensors['input']

    if input_array.shape[0] != 1:
        raise ValueError("Request has to have a single sample")

    x: np.ndarray = input_array[0]

    # Create temporary servable, so main servable won't be affected
    temp_servable_copy: HydroServingServable = hs_client.deploy_servable(model_name, model_version)
    logger.info("Servable deployed")

    # Convert config dicts to appropriate types
    # label_decoders: Dict[int, List[str]] = dict([(int(k), v) for (k, v) in config['label_decoders'].items()])
    label_decoders: Dict[int, List[str]] = dict()  # This is a temp workaround, since we can't pass this config through UI

    # oh_encoded_categories: Dict[str, List[int]] = dict(
    #     [(k, [int(v) for v in vs]) for (k, vs) in config['oh_encoded_categories'].items()])
    oh_encoded_categories: Dict[str, List[int]] = dict()  # This is a temp workaround, since we can't pass this config through UI

    anchor_explainer = TabularExplainer()

    # Get subsample to work with
    logger.info("Start fetching reqstore data")
    reqstore_data = utils.get_tensor_subsample(rs_client, model, config.get("subsample_size", 5000))
    logger.info("Finished fetching reqstore data")

    anchor_explainer.fit(data=reqstore_data,
                         label_decoders=label_decoders,
                         ordinal_features_idx=config.get('ordinal_features_idx', [0, 11]),
                         oh_encoded_categories=oh_encoded_categories,
                         feature_names=None)

    # FIXME add classifier_fn support for contracts without batch dim
    explanation = anchor_explainer.explain(x, classifier_fn=lambda k: temp_servable_copy(k)['classes'])

    result_json = {"explanation": [str(p) for p in explanation.predicates],
                   "coverage": explanation.coverage(),
                   "precision": explanation.precision()}

    # Store explanation in MongoDB 'root_cause' db
    db.anchor_explanations.update_one({"_id": explanation_id}, {"$set": {'result': result_json,
                                                                         "completed_at": datetime.datetime.now()}})

    logger.info(f"Finished task to explain {model_name}_{model_version} with anchor")

    # Remove temporary servable
    temp_servable_copy.delete()

    return str(explanation_id)
