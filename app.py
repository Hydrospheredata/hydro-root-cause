import datetime
import os
import time
from functools import partial
from typing import Dict, List

from bson import objectid
import numpy as np
import pandas as pd
import requests
from anchor2 import TabularExplainer
from celery import Celery
from dotenv import load_dotenv
from flask import Flask, request, jsonify, url_for
from hydro_serving_grpc.timemachine import ReqstoreClient
from jsonschema import validate
from pymongo import MongoClient
from pymongo.database import Database

from rise.rise import RiseImageExplainer

load_dotenv()
REQSTORE_URL = os.getenv("REQSTORE_URL")
SERVING_URL = os.getenv("SERVING_URL", "http://fc13d681.serving.odsc.k8s.hydrosphere.io")
MONGO_URL = os.getenv("MONGO_URL", "mongo")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
REDIS_URL = os.getenv("REDIS_URL", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

mongo_client = MongoClient(host=MONGO_URL, port=MONGO_PORT)
db = mongo_client['root_cause']

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = f"redis://{REDIS_URL}:{REDIS_PORT}/0"
app.config['CELERY_RESULT_BACKEND'] = f"redis://{REDIS_URL}:{REDIS_PORT}/0"

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
        raise NotImplementedError

    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': "PENDING",
        }
    elif task.state == 'SUCCESS':
        # job completed
        response = {
            'state': task.state,
            'result': task.result
        }
    elif task.state == "STARTED":
        # job is in progress
        response = {
            'state': task.state,
            'progress': task.info['progress']
        }
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'status': str(task.info),  # this is the exception raised
        }

    return jsonify(response)


# Function to store sample in a json with columnar signature
def make_columnar_json(sample, feature_names):
    # TODO remove columnar json to row-based json
    output_json = {}
    if type(sample) == pd.Series:
        feature_names = sample.index
        sample = sample.values
    elif feature_names is None:
        raise ValueError

    for feature_idx, fname in enumerate(feature_names):
        output_json[fname] = [int(v) for v in sample[:, feature_idx]]
    return output_json


def get_application_id(application_name: str):
    r = requests.get(f"{SERVING_URL}/api/v2/application/{application_name}")
    return r.json()['executionGraph']['stages'][0]['modelVariants'][0]['modelVersion']["id"]


def get_data_from_reqstore(application_name):
    # TODO change to subsampling Dima
    def request_to_df(r):
        columns = []
        values = []
        for key, value in r.inputs.items():
            columns.append(key)
            values.append(value.int64_val)
        return pd.DataFrame(columns=columns, data=np.array(values).T)

    client = ReqstoreClient(REQSTORE_URL, False)
    end_time = round(time.time() * 1000000000)
    application_id = get_application_id(application_name)
    data = client.getRange(0, end_time, application_id)
    data = list(data)

    rs = list(map(lambda x: x.entries[0].request, data))
    rs = [request_to_df(r) for r in rs]

    df = pd.concat(rs, sort=False)

    # Remove [1, 1, 1, ...] which results from UI test model calls
    df = df.loc[~np.all(df == np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), axis=1)]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    return df


def hydroserving_classifier_call(x, feature_names, application_name, return_label="Prediction"):
    # TODO change to dirka Bulata
    response = requests.post(url=f"{SERVING_URL}/gateway/application/{application_name}",
                             json=make_columnar_json(x, feature_names=feature_names))
    predicted_label = np.array(response.json()[return_label])
    return predicted_label


def hydroserving_image_classifier_call(x, application_name):
    # TODO change to dirka Bulata
    url = f"{SERVING_URL}/gateway/application/{application_name}"
    print(url)
    response = requests.post(url=url, json={"imgs": x.tolist()})
    try:
        predicted_probas = np.array(response.json()["probabilities"])
        return predicted_probas
    except KeyError:
        print("Probabilities not found", response.text)
        raise ValueError
    except ValueError:
        print(response.text)
        raise ValueError


# ----------------------------------------ANCHOR---------------------------------------- #

anchor_schema = {
    "type": "object",
    "properties": {
        "application_name": {"type": "string"},
        "config": {"type": "object",
                   "properties": {
                       "feature_names": {"type": "array"},
                       "ordinal_features_idx": {"type": "array"},
                       "label_decoders": {"type": "object"},
                       "oh_encoded_categories": {"type": "object"},
                   }
                   },
    },
}


@app.route("/anchor", methods=['POST'])
def anchor():
    #  TODO if mongo anchor cache is available we can check anchor presence by binary mask
    #  ex: db.collection.save({ _id: 1, a: 54, binaryValueofA: "00110110" })
    #  ex: db.collection.find( { a: { $bitsAllSet: [ 1, 5 ] } } )

    inp_json = request.get_json()
    validate(inp_json, anchor_schema)

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

    application_name = job_json['application_name']
    config = job_json['config']
    x = np.array(job_json['explained_instance'])

    # Convert config dicts to appropriate types
    label_decoders: Dict[int, List[str]] = dict([(int(k), v) for (k, v) in config['label_decoders'].items()])
    oh_encoded_categories: Dict[str, List[int]] = dict(
        [(k, [int(v) for v in vs]) for (k, vs) in config['oh_encoded_categories'].items()])

    anchor_explainer = TabularExplainer()

    data = get_data_from_reqstore(application_name)
    data = data.loc[:, config['feature_names']]  # Sort columns according to feature names order

    anchor_explainer.fit(data=data,
                         label_decoders=label_decoders,
                         ordinal_features_idx=config['ordinal_features_idx'],
                         oh_encoded_categories=oh_encoded_categories,
                         feature_names=config['feature_names'])

    classifier_fn = partial(hydroserving_classifier_call,
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


rise_schema = {
    "type": "object",
    "properties": {
        "application_name": {"type": "string"},
        "config": {"type": "object",
                   "properties": {
                       "input_size": {"type": "array"},
                       "number_of_masks": {"type": "number"},
                       "mask_granularity": {"type": "number"},
                       "mask_density": {"type": "number"},
                       "single_channel": {"type": "boolean"},
                   }
                   },
    },
}


@app.route("/rise", methods=['POST'])
def rise():
    inp_json = request.get_json()
    validate(inp_json, rise_schema)
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

    application_name = job_json['application_name']
    rise_config = job_json['config']
    image = np.array(job_json['image'])

    rise_explainer = RiseImageExplainer()
    prediction_fn = partial(hydroserving_image_classifier_call, application_name=application_name)
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

    result_json = {"masks": saliency_map.tostring(),
                   "shape": saliency_map.shape}

    db.rise_explanations.update_one({"_id": explanation_id}, {"$set": {'result': result_json,
                                                                       "completed_at": datetime.datetime.now()}})

    return str(explanation_id)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
