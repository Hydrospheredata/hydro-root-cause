import datetime
import os
import time
from functools import partial
from typing import Dict, List

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

from rise.rise import RiseImageExplainer

load_dotenv()
REQSTORE_URL = os.getenv("REQSTORE_URL")
SERVING_URL = os.getenv("SERVING_URL")
MONGO_URL = os.getenv("MONGO_URL", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
REDIS_URL = os.getenv("REDIS_URL", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

db = MongoClient(host=MONGO_URL, port=MONGO_PORT)
app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


@app.route("/")
def hello():
    return "Hello!"


# Function to store sample in a json with columnar signature
def make_columnar_json(sample, feature_names):
    output_json = {}
    if type(sample) == pd.Series:
        feature_names = sample.index
        sample = sample.values
    elif feature_names is None:
        raise ValueError

    for feature_idx, fname in enumerate(feature_names):
        output_json[fname] = [int(v) for v in sample[:, feature_idx]]
    return output_json


def get_data_from_reqstore(application_id, reqstore_url=REQSTORE_URL):
    def request_to_df(r):
        columns = []
        values = []
        for key, value in r.inputs.items():
            columns.append(key)
            values.append(value.int64_val)
        return pd.DataFrame(columns=columns, data=np.array(values).T)

    client = ReqstoreClient(reqstore_url, False)
    end_time = round(time.time() * 1000000000)
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


def get_application_id(application_name: str):
    service_link = SERVING_URL
    r = requests.get(service_link + application_name)
    return r.json()['executionGraph']['stages'][0]['modelVariants'][0]['modelVersion']["id"]


def hydroserving_classifier_call(x, feature_names, application_name,
                                 service_link="https://dev.k8s.hydrosphere.io/gateway/application/",
                                 return_label="Prediction"):
    response = requests.post(url=service_link + application_name, json=make_columnar_json(x, feature_names=feature_names))
    predicted_label = np.array(response.json()[return_label])
    return predicted_label


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
    #  if ...:
    #
    #  db.collection.save({ _id: 1, a: 54, binaryValueofA: "00110110" })
    #  db.collection.find( { a: { $bitsAllSet: [ 1, 5 ] } } )

    inp_json = request.get_json()
    validate(inp_json, anchor_schema)
    application_id = get_application_id(inp_json['application_name'])
    anchor_config = inp_json['config']
    data = get_data_from_reqstore(str(application_id))
    print("DATA SHAPE ", data.shape)

    label_decoders: Dict[int, List[str]] = dict([(int(k), v) for (k, v) in anchor_config['label_decoders'].items()])
    oh_encoded_categories: Dict[str, List[int]] = dict(
        [(k, [int(v) for v in vs]) for (k, vs) in anchor_config['oh_encoded_categories'].items()])

    anchor_explainer = TabularExplainer()
    data = data.loc[:, anchor_config['feature_names']]  # Sort columns according to feature names order
    anchor_explainer.fit(data=data,
                         label_decoders=label_decoders,
                         ordinal_features_idx=anchor_config['ordinal_features_idx'],
                         oh_encoded_categories=oh_encoded_categories,
                         feature_names=anchor_config['feature_names'])

    x = np.array(request.get_json()['explained_instance'])

    classifier_fn = partial(hydroserving_classifier_call,
                            feature_names=anchor_config['feature_names'],
                            application_name=inp_json['application_name'])

    explanation = anchor_explainer.explain(x, classifier_fn=classifier_fn)

    return jsonify(explanation=str(explanation), coverage=explanation.coverage(), precision=explanation.precision())


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

    rise_explanation_id = db.rise_explanations.insert(inp_json)
    task = rise_task.delay(rise_explanation_id)

    return jsonify({}), 202, {'Location': url_for('task_status', task_id=task.id)}


@app.route('/status/<task_id>', methods=["GET"])
def task_status(task_id):
    task = rise_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'status': str(task.info),  # this is the exception raised
        }

    return jsonify(response)


def hydroserving_image_classifier_call(x,
                                       application_name,
                                       service_link=SERVING_URL):
    url = service_link + application_name
    response = requests.post(url=url, json={"imgs": x.tolist()})
    predicted_probas = np.array(response.json()["probabilities"])
    return predicted_probas


@celery.task(bind=True)
def rise_task(self, explanation_id):
    job_json = db.rise_explanations.find_one({"_id": explanation_id})

    if 'result' in job_json:
        return job_json['results']

    application_name = job_json['application_name']
    rise_config = job_json['rise_config']
    image = np.array(job_json['image'])

    rise_explainer = RiseImageExplainer()
    prediction_fn = partial(hydroserving_image_classifier_call, application_name=application_name)
    rise_explainer.fit(prediction_fn=prediction_fn,
                       input_size=rise_config['input_size'],
                       number_of_masks=rise_config['number_of_masks'],
                       mask_granularity=rise_config['mask_granularity'],
                       mask_density=rise_config['mask_density'],
                       single_channel=rise_config['single_channel'])

    saliency_map = rise_explainer.explain(image,
                                          state_updater=lambda x: self.update_state(state='PROGRESS', meta={'progress': x}))

    result_json = {"masks": saliency_map.tostring(),
                   "shape": saliency_map.shape}

    db.rise_explanations.update_one({"_id": explanation_id}, {"$set": {'result': result_json}})

    return explanation_id


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
