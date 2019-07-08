import datetime
import os
from functools import partial
from typing import Dict, List

import numpy as np
from bson import objectid
from celery import Celery
from flask import Flask, request, jsonify, url_for
from jsonschema import validate
from pymongo import MongoClient
from pymongo.database import Database

import utils
from anchor2 import TabularExplainer
from rise.rise import RiseImageExplainer

REQSTORE_URL = os.getenv("REQSTORE_URL")
SERVING_URL = os.getenv("SERVING_URL", "http://fc13d681.serving.odsc.k8s.hydrosphere.io")
MONGO_URL = os.getenv("MONGO_URL", "mongo")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))

mongo_client = MongoClient(host=MONGO_URL, port=MONGO_PORT)
db = mongo_client['root_cause']

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = f"mongodb://{MONGO_URL}:{MONGO_PORT}/celery_broker'"
app.config['CELERY_RESULT_BACKEND'] = f"mongodb://{MONGO_URL}:{MONGO_PORT}/celery_backend"

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
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
        response['result'] = url_for('fetch_result', _external=True, result_id=str(task.result), method=method)

    elif task.state == "STARTED":
        # job is in progress, return progress
        response['progress'] = task.info['progress']

    else:
        # something went wrong in the background job, return the exception raised
        response['status'] = str(task.info),

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


root_cause_schema = {
    "type": "object",
    "properties": {
        "servable_name": {"type": "string"},
        "explained_instance": {"type": "object"}
    },
}


# ----------------------------------------ANCHOR---------------------------------------- #


@app.route("/anchor", methods=['POST'])
def anchor():
    #  TODO if mongo anchor cache is available we can check anchor presence by binary mask
    #  ex: db.collection.save({ _id: 1, a: 54, binaryValueofA: "00110110" })
    #  ex: db.collection.find( { a: { $bitsAllSet: [ 1, 5 ] } } )

    inp_json = request.get_json()
    validate(inp_json, root_cause_schema)

    inp_json['created_at'] = datetime.datetime.now()

    anchor_explanation_id = db.anchor_explanations.insert_one(inp_json).inserted_id
    task = anchor_task.delay(str(anchor_explanation_id))

    return jsonify({}), 202, {'Location': url_for('task_status', task_id=task.id, method="anchor")}


@celery.task(bind=True)
def anchor_task(explanation_id: str):
    # Shadows names from flask server scope to prevent problems with forks
    mongo_client: MongoClient = MongoClient(host=MONGO_URL, port=MONGO_PORT)
    db: Database = mongo_client['root_cause']

    explanation_id = objectid.ObjectId(explanation_id)
    job_json = db.anchor_explanations.find_one_and_update({"_id": explanation_id},
                                                          {"$set": {"started_at": datetime.datetime.now()}})

    if 'result' in job_json:
        return job_json['results']

    # TODO deploy servable

    application_name = job_json['application_name']
    config = job_json['config']
    x = np.array(job_json['explained_instance'])

    # Convert config dicts to appropriate types
    # label_decoders: Dict[int, List[str]] = dict([(int(k), v) for (k, v) in config['label_decoders'].items()])
    label_decoders: Dict[int, List[str]] = dict()

    # oh_encoded_categories: Dict[str, List[int]] = dict(
    #     [(k, [int(v) for v in vs]) for (k, vs) in config['oh_encoded_categories'].items()])
    oh_encoded_categories: Dict[str, List[int]] = dict()

    anchor_explainer = TabularExplainer()

    data = utils.get_data_from_reqstore(application_name)
    data = data.loc[:, config['feature_names']]  # Sort columns according to feature names order

    anchor_explainer.fit(data=data,
                         label_decoders=label_decoders,
                         ordinal_features_idx=config['ordinal_features_idx'],
                         oh_encoded_categories=oh_encoded_categories,
                         feature_names=config['feature_names'])

    classifier_fn = partial(utils.hs_call,
                            feature_names=config['feature_names'],
                            application_name=application_name)

    explanation = anchor_explainer.explain(x, classifier_fn=classifier_fn)

    result_json = {"explanation": str(explanation),
                   "coverage": explanation.coverage(),
                   "precision": explanation.precision()}

    db.anchor_explanations.update_one({"_id": explanation_id}, {"$set": {'result': result_json,
                                                                         "completed_at": datetime.datetime.now()}})

    return str(explanation_id)


# ----------------------------------------RISE---------------------------------------- #


@app.route("/rise", methods=['POST'])
def rise():
    inp_json = request.get_json()
    validate(inp_json, root_cause_schema)
    inp_json['created_at'] = datetime.datetime.now()

    rise_explanation_id = db.rise_explanations.insert_one(inp_json).inserted_id
    task = rise_task.delay(str(rise_explanation_id))

    return jsonify({}), 202, {'Location': url_for('task_status', task_id=task.id, method="rise")}


@celery.task(bind=True)
def rise_task(self, explanation_id: str):
    # Shadows names from flask server scope to prevent problems with forks
    mongo_client: MongoClient = MongoClient(host=MONGO_URL, port=MONGO_PORT)
    db: Database = mongo_client['root_cause']

    explanation_id = objectid.ObjectId(explanation_id)
    job_json = db.rise_explanations.find_one_and_update({"_id": explanation_id},
                                                        {"$set": {"started_at": datetime.datetime.now()}})

    if 'result' in job_json:
        return job_json['results']

    # TODO deploy servable

    application_name = job_json['application_name']
    image = np.array(job_json['explained_instance'])

    input_dims = (28, 28)  # TODO get input size from contract
    rise_config = {"input_size": input_dims,
                   "number_of_masks": 1000,
                   "mask_granularity": 0.3,
                   "mask_density": 0.5,
                   "single_channel": False}

    rise_explainer = RiseImageExplainer()
    prediction_fn = partial(utils.hs_img_call, application_name=application_name)
    rise_explainer.fit(prediction_fn=prediction_fn,
                       input_size=rise_config['input_size'],
                       number_of_masks=rise_config['number_of_masks'],
                       mask_granularity=rise_config['mask_granularity'],
                       mask_density=rise_config['mask_density'],
                       single_channel=rise_config['single_channel'])

    def state_updater(x):
        self.update_state(state='STARTED', meta={'progress': x})

    saliency_map: np.array = rise_explainer.explain(image,
                                                    state_updater=state_updater)

    result_json = {"masks": saliency_map.tolist()}

    db.rise_explanations.update_one({"_id": explanation_id}, {"$set": {'result': result_json,
                                                                       "completed_at": datetime.datetime.now()}})

    return str(explanation_id)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
