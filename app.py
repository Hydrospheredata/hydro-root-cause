import datetime
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from anchor2 import TabularExplainer
from bson import objectid
from celery import Celery
from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
from hydro_serving_grpc.reqstore import reqstore_client
from loguru import logger
from pymongo import MongoClient
from pymongo.database import Database

from client import HydroServingClient, HydroServingServable
from rise.rise import RiseImageExplainer

REQSTORE_URL = os.getenv("REQSTORE_URL", "managerui:9090")
SERVING_URL = os.getenv("SERVING_URL", "managerui:9090")

MONGO_URL = os.getenv("MONGO_URL", "mongodb")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")

mongo_client = MongoClient(host=MONGO_URL, port=MONGO_PORT, maxPoolSize=200,
                           username=MONGO_USER, password=MONGO_PASS,
                           authSource=MONGO_AUTH_DB)
db = mongo_client['root_cause']

hs_client = HydroServingClient(SERVING_URL)
rs_client = reqstore_client.ReqstoreClient(REQSTORE_URL, insecure=True)

app = Flask(__name__)

CORS(app, expose_headers=['location'])

connection_string = f"mongodb://{MONGO_URL}:{MONGO_PORT}"
if MONGO_USER is not None and MONGO_PASS is not None:
    connection_string = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_URL}:{MONGO_PORT}"
app.config['CELERY_BROKER_URL'] = f"{connection_string}/celery_broker"
app.config['CELERY_RESULT_BACKEND'] = f"{connection_string}/celery_backend"

celery = Celery(app.name,
                broker=app.config['CELERY_BROKER_URL'],
                backend=app.config['CELERY_RESULT_BACKEND'])
celery.conf.update(app.config)


@app.route("/")
def hello():
    return "hydro_root_cause_service"


@app.route('/status/<method>/<task_id>', methods=["GET"])
def task_status(task_id, method):
    if method == "rise":
        task = rise_task.AsyncResult(task_id)
    elif method == "anchor":
        task = anchor_task.AsyncResult(task_id)
    else:
        raise ValueError

    response = {
        'state': task.state,
    }

    if task.state == 'PENDING':
        # job did not start yet, do nothing
        pass
    elif task.state == 'SUCCESS':
        # job completed, return url to result
        response['result'] = str(task.result)

    elif task.state == "STARTED":
        # job is in progress, return progress
        response['progress'] = task.info['progress']

    else:
        # something went wrong in the background job, return the exception raised
        response['description'] = str(task.info)

    return jsonify(response)


@app.route('/fetch_result/<method>/<result_id>', methods=["GET"])
def fetch_result(result_id, method):
    if method == "rise":
        collection = db.rise_explanations
    elif method == "anchor":
        collection = db.anchor_explanations
    else:
        raise ValueError

    explanation = collection.find_one({"_id": objectid.ObjectId(result_id)})
    del explanation['_id']

    if method == "rise":
        explanation['result']['masks'] = str(explanation['result']['masks'])

    return jsonify(explanation)


# ----------------------------------------ANCHOR---------------------------------------- #

@app.route("/anchor", methods=['POST'])
def anchor():
    #  TODO if mongo anchor cache is available we can check anchor presence by binary mask
    #  ex: db.collection.save({ _id: 1, a: 54, binaryValueofA: "00110110" })
    #  ex: db.collection.find( { a: { $bitsAllSet: [ 1, 5 ] } } )

    inp_json = request.get_json()
    logger.info(f"Received request to explain {str(inp_json['model'])} with anchor")

    inp_json['created_at'] = datetime.datetime.now()

    anchor_explanation_id = db.anchor_explanations.insert_one(inp_json).inserted_id
    task = anchor_task.delay(str(anchor_explanation_id))

    return jsonify({}), 202, {'Location': '/rootcause' + url_for('task_status', task_id=task.id, _external=False, method="anchor")}


@celery.task(bind=True)
def anchor_task(self, explanation_id: str):
    # Shadows names from flask server scope to prevent problems with forks
    mongo_client: MongoClient = MongoClient(host=MONGO_URL, port=MONGO_PORT,
                                            username=MONGO_USER, password=MONGO_PASS, authSource=MONGO_AUTH_DB)
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

    input_tensors = get_reqstore_request_tensors(model.contract, rs_client, folder, ts, reqstore_uid)

    if len(input_tensors) != 1:
        raise ValueError("Request has to many input tensors")

    x: np.ndarray = list(input_tensors.values())[0]

    logger.info(
        f"Initiated task to explain image folder:{folder}, ts:{ts}, uid:{reqstore_uid} under model {model_name}_{model_version} with rise")

    # Create temporary servable, so main servable won't be affected
    temp_servable_copy: HydroServingServable = hs_client.deploy_servable(model_name, model_version)

    # Convert config dicts to appropriate types
    # label_decoders: Dict[int, List[str]] = dict([(int(k), v) for (k, v) in config['label_decoders'].items()])
    label_decoders: Dict[int, List[str]] = dict()  # This is a temp workaround, since we can't pass this config through UI

    # oh_encoded_categories: Dict[str, List[int]] = dict(
    #     [(k, [int(v) for v in vs]) for (k, vs) in config['oh_encoded_categories'].items()])
    oh_encoded_categories: Dict[str, List[int]] = dict()  # This is a temp workaround, since we can't pass this config through UI

    anchor_explainer = TabularExplainer()

    # Get subsample to work with
    model_id = temp_servable_copy.id
    rs_entries = []
    for r in rs_client.subsampling(str(model_id), amount=config.get("subsample_size", 5000)):
        rs_entries.extend(r.entries)

    requests = [model.contract.decode_request(e.request) for e in rs_entries]

    # concat requests into dataframe format
    rs = []
    feature_order = model.contract.input_names
    for r in requests:
        column_arrays = []
        for feature_name in feature_order:
            column_arrays.append(r[feature_name])
        df = pd.DataFrame(np.hstack(column_arrays), columns=feature_order)
        rs.append(df)
    reqstore_data = pd.concat(rs)

    # Sort columns according to feature names order
    reqstore_data = reqstore_data.loc[:, model.contract.input_names]

    anchor_explainer.fit(data=reqstore_data,
                         label_decoders=label_decoders,
                         ordinal_features_idx=config['ordinal_features_idx'],
                         oh_encoded_categories=oh_encoded_categories,
                         feature_names=model.contract.input_names)

    explanation = anchor_explainer.explain(x, classifier_fn=temp_servable_copy)

    result_json = {"explanation": str(explanation),
                   "coverage": explanation.coverage(),
                   "precision": explanation.precision()}

    # Store explanation in MongoDB 'root_cause' db
    db.anchor_explanations.update_one({"_id": explanation_id}, {"$set": {'result': result_json,
                                                                         "completed_at": datetime.datetime.now()}})

    logger.info(f"Finished task to explain {model_name}_{model_version} with anchor")

    # Remove temporary servable
    temp_servable_copy.delete()

    return str(explanation_id)


# ----------------------------------------RISE---------------------------------------- #


@app.route("/rise", methods=['POST'])
def rise():
    inp_json = request.get_json()
    inp_json['created_at'] = datetime.datetime.now()

    logger.info(f"Received request to explain {str(inp_json['model'])} with rise")

    rise_explanation_id = db.rise_explanations.insert_one(inp_json).inserted_id
    task = rise_task.delay(str(rise_explanation_id))

    return jsonify({}), 202, {'Location': '/rootcause' + url_for('task_status', _external=False, task_id=task.id, method="rise")}


def get_reqstore_request_tensors(model_contract, rclient: reqstore_client.ReqstoreClient, folder, timestamp, uid: int):
    request_response: reqstore_client.TsRecord = rclient.get(folder, timestamp, uid)
    request = request_response.entries[0].request
    input_tensors = model_contract.decode_request(request)
    return input_tensors


@celery.task(bind=True)
def rise_task(self, explanation_id: str):
    # Shadows names from flask server scope to prevent problems with forks
    mongo_client: MongoClient = MongoClient(host=MONGO_URL, port=MONGO_PORT,
                                            username=MONGO_USER, password=MONGO_PASS, authSource=MONGO_AUTH_DB)
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

    input_tensors = get_reqstore_request_tensors(model.contract, rs_client, folder, ts, reqstore_uid)

    if len(input_tensors) != 1:
        raise ValueError("Request has to many input tensors")

    image: np.ndarray = list(input_tensors.values())[0]

    if len(image.shape) > 4:
        raise ValueError(f"Invalid image shape: {image.shape}. Expected 2, 3 or 4 dimensions")
    elif len(image.shape) == 4:
        image = image[0]

    # if len(image.shape) == 3 and (image.shape[-1] != 3 and image.shape[-1] != 1):
    #     raise ValueError(f"Invalid image shape: {image.shape}. Expected channels_last")

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
    min = np.min(saliency_map)
    max = np.max(saliency_map)
    saliency_map = (saliency_map - min) / (max - min)

    np.rint(saliency_map * 255, out=saliency_map)  # Round to int pixel values
    saliency_map = saliency_map.astype(np.uint8)  # Use corresponding dtype

    result_json = {"masks": saliency_map.tolist()}

    # Store explanation in MongoDB
    db.rise_explanations.update_one({"_id": explanation_id}, {"$set": {'result': result_json,
                                                                       "completed_at": datetime.datetime.now()}})

    logger.info(f"Finished task to explain {model_name}_{model_version} with rise")

    temp_servable_copy.delete()
    return str(explanation_id)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
