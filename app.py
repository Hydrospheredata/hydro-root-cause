import datetime
import json
import logging
import os
import sys
from enum import Enum, auto
from logging.config import fileConfig

import pymongo
import requests
from bson import objectid
from celery import Celery
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from hydrosdk.cluster import Cluster
from jsonschema import Draft7Validator
from pymongo import MongoClient
from waitress import serve

fileConfig("logging_config.ini")


class ExplanationState(Enum):
    NOT_CALLED = auto()
    PENDING = auto()
    STARTED = auto()
    SUCCESS = auto()
    FAILED = auto()
    NOT_SUPPORTED = auto()


with open("version.json") as version_file:
    BUILDINFO = json.load(version_file)  # Load buildinfo with branchName, headCommitId and version label
    BUILDINFO['pythonVersion'] = sys.version  # Augment with python runtime version

DEBUG_ENV = bool(os.getenv("DEBUG", True))

HS_CLUSTER_ADDRESS = os.getenv("HTTP_UI_ADDRESS")
MONITORING_URL = f"{HS_CLUSTER_ADDRESS}/monitoring"

GRPC_ADDRESS = os.getenv("GRPC_UI_ADDRESS")

S3_ENDPOINT = os.getenv("S3_ENDPOINT")

MONGO_URL = os.getenv("MONGO_URL")
MONGO_PORT = int(os.getenv("MONGO_PORT"))

MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")

with open("json_schemas/explanation_request.json") as f:
    REQUEST_JSON_SCHEMA = json.load(f)
    validator = Draft7Validator(REQUEST_JSON_SCHEMA)


def get_mongo_client():
    return MongoClient(host=MONGO_URL, port=MONGO_PORT, maxPoolSize=200,
                       username=MONGO_USER, password=MONGO_PASS,
                       authSource=MONGO_AUTH_DB)


mongo_client = get_mongo_client()
db = mongo_client['root_cause']

hs_cluster = Cluster(HS_CLUSTER_ADDRESS)

app = Flask(__name__)

CORS(app, expose_headers=['location'])

connection_string = f"mongodb://{MONGO_URL}:{MONGO_PORT}"
if MONGO_USER is not None and MONGO_PASS is not None:
    connection_string = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_URL}:{MONGO_PORT}"

app.config['CELERY_BROKER_URL'] = f"{connection_string}/celery_broker?authSource={MONGO_AUTH_DB}"
app.config['CELERY_RESULT_BACKEND'] = f"{connection_string}/celery_backend?authSource={MONGO_AUTH_DB}"

celery = Celery(app.name,
                broker=app.config['CELERY_BROKER_URL'],
                backend=app.config['CELERY_RESULT_BACKEND'])

celery.autodiscover_tasks(["rise_tasks", "anchor_tasks"], force=True)
celery.conf.update(app.config)
celery.conf.update({"CELERY_DISABLE_RATE_LIMITS": True})

import utils
import anchor_tasks
import rise_tasks

TASKS = {"anchor": anchor_tasks.tasks.anchor_task}


@app.route("/", methods=['GET'])
def hello():
    return "Hi! I am RootCause Service"


@app.route("/buildinfo", methods=['GET'])
def buildinfo():
    return jsonify(BUILDINFO)


@app.route('/task_status', methods=["GET"])
def get_celery_task_status():
    possible_args = {"method", "task_id"}
    if set(request.args.keys()) != possible_args:
        return jsonify({"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400

    method = request.args['method']
    task_id = request.args['task_id']

    task = TASKS[method].AsyncResult(task_id)

    response = {
        'state': task.state,
        'task_id': task_id
    }
    return jsonify(response)


@app.route('/explanation', methods=["GET"])
def get_explanation_status():
    possible_args = {"model_version_id", "explained_request_id", "method"}
    if set(request.args.keys()) != possible_args:
        return jsonify({"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400

    method = request.args['method']
    request_id = request.args['explained_request_id']
    model_version_id = int(request.args['model_version_id'])

    model_supported, support_description = utils.check_model_version_support(db, method, model_version_id)
    if not model_supported:
        return jsonify({"state": ExplanationState.NOT_SUPPORTED.name,
                        "description": support_description})

    # TODO check task state - if it is "FAILED", and state in mongodb is not - discard mongodb state and description, and return FAILED
    explanation = db[method].find_one({"model_version_id": model_version_id, "explained_request_id": request_id})
    if explanation:
        response = {"state": explanation['state'],
                    "description": explanation['description']}
        if 'result' in explanation:
            response['result'] = explanation['result']
        return jsonify(response)
    else:
        return jsonify({"state": ExplanationState.NOT_CALLED.name,
                        "description": "Explanation was never requested"})


@app.route("/explanation", methods=["POST"])
def calculate_new_explanation():
    inp_json = request.get_json()

    if not validator.is_valid(inp_json):
        error_message = "\n".join([e.message for e in validator.iter_errors(inp_json)])
        return jsonify({"message": error_message}), 400

    model_version_id = inp_json['model_version_id']
    explained_request_id = inp_json['explained_request_id']
    method = inp_json['method']

    has_training_data = len(requests.get(f"{MONITORING_URL}/training_data?modelVersionId={model_version_id}").json()) > 0
    model_supported, support_description = utils.check_model_version_support(db, method, model_version_id)
    if not model_supported:
        return jsonify({"state": ExplanationState.NOT_SUPPORTED.name,
                        "description": support_description})

    if method == "anchor":
        task = anchor_tasks.tasks.anchor_task
    elif method == "rise":
        task = rise_tasks.tasks.rise_task
    else:
        return 400, f"Invalid method: '{method}'"

    logging.info(f"Received request to explain request={explained_request_id} of model {model_version_id} with {method}")

    # Check for existing explanations requests for this method, model and request.
    explanation = db[method].find_one({"model_version_id": model_version_id,
                                       "explained_request_id": explained_request_id},
                                      sort=[("_id", pymongo.DESCENDING)])
    if explanation:
        logging.info(f"Found existing request for {method} explanation for {model_version_id} - {explained_request_id}")
        response = {"state": explanation['state'],
                    "description": explanation['description']}
        if 'result' in explanation:
            response['result'] = explanation['result']
        return jsonify(response)

    explanation_job_description = {'created_at': datetime.datetime.now(),
                                   'model_version_id': model_version_id,
                                   'explained_request_id': explained_request_id,
                                   "state": ExplanationState.PENDING.name,
                                   "description": "Received explanation request",
                                   'method': method}

    explanation_id = db[method].insert_one(explanation_job_description).inserted_id
    logging.info(f"Stored request into mongodb for {method} explanation for {model_version_id} - {explained_request_id}")

    task_async_result = task.delay(str(explanation_id),
                                   queue='rootcause')

    explanation_id = objectid.ObjectId(explanation_id)
    db[method].find_one_and_update({"_id": explanation_id},
                                   {"$set": {"celery_task_id": task_async_result.id,
                                             "description": "Explanation is queued"}})

    return jsonify({}), 202


@app.route("/config", methods=['GET', 'PATCH'])
def get_params():
    possible_args = {"model_version_id", "method"}
    if set(request.args.keys()) != possible_args:
        return jsonify({"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400

    method = request.args['method']
    model_version_id = int(request.args['model_version_id'])

    current_config = utils.get_latest_config(db, method, model_version_id)

    if request.method == 'GET':
        return jsonify(current_config)
    elif request.method == "PATCH":
        inp_json = request.get_json()

        # Patch current config with new values from inp_json
        new_config = {**current_config, **inp_json['config']}

        # Save patched version of config
        db["config"].insert_one({"method": method,
                                 'created_at': datetime.datetime.now(),
                                 "config": new_config,
                                 "model_version_id": model_version_id})
        return Response(status=200)
    else:
        return Response(status=405)


if __name__ == "__main__":
    if not DEBUG_ENV:
        serve(app, host='0.0.0.0', port=5000)
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)
