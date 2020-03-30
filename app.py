import datetime
import json
import logging
import os
import sys
from logging.config import fileConfig

import requests
from bson import objectid
from celery import Celery
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from hydrosdk.cluster import Cluster
from hydrosdk.model import Model
from jsonschema import Draft7Validator
from pymongo import MongoClient
from waitress import serve

import utils

fileConfig("logging_config.ini")

with open("version.json") as version_file:
    BUILDINFO = json.load(version_file)  # Load buildinfo with branchName, headCommitId and version label
    BUILDINFO['pythonVersion'] = sys.version  # Augment with python runtime version

DEBUG_ENV = bool(os.getenv("DEBUG_ENV", True))

# TODO return before merge into master
# SERVING_URL = os.getenv("SERVING_URL", "httplocalhost")
# MONITORING_URL = os.getenv("MONITORING_URL")

SERVING_URL = "http://localhost"
MONITORING_URL = "http://localhost/monitoring"

MONGO_URL = os.getenv("MONGO_URL", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")

CONFIG_COLLECTION = os.getenv("CONFIG_COLLECTION_NAME", "config")

with open("json_schemas/explanation_request.json") as f:
    REQUEST_JSON_SCHEMA = json.load(f)
    validator = Draft7Validator(REQUEST_JSON_SCHEMA)


def get_mongo_client():
    return MongoClient()
    # TODO return before merge into master
    # return MongoClient(host=MONGO_URL, port=MONGO_PORT, maxPoolSize=200,
    #                    username=MONGO_USER, password=MONGO_PASS,
    #                    authSource=MONGO_AUTH_DB)


mongo_client = get_mongo_client()
db = mongo_client['root_cause']

hs_cluster = Cluster(SERVING_URL)

app = Flask(__name__)

CORS(app, expose_headers=['location'])

connection_string = f"mongodb://{MONGO_URL}:{MONGO_PORT}"
# if MONGO_USER is not None and MONGO_PASS is not None:
#     connection_string = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_URL}:{MONGO_PORT}"

app.config['CELERY_BROKER_URL'] = f"{connection_string}/celery_broker?authSource={MONGO_AUTH_DB}"
app.config['CELERY_RESULT_BACKEND'] = f"{connection_string}/celery_backend?authSource={MONGO_AUTH_DB}"

celery = Celery(app.name,
                broker=app.config['CELERY_BROKER_URL'],
                backend=app.config['CELERY_RESULT_BACKEND'])

celery.autodiscover_tasks(["rise_tasks", "anchor_tasks"], force=True)
celery.conf.update(app.config)
celery.conf.update({"CELERY_DISABLE_RATE_LIMITS": True})

import anchor_tasks
import rise_tasks


@app.route("/", methods=['GET'])
def hello():
    return "Hi! I am RootCause Service"


@app.route("/buildinfo", methods=['GET'])
def buildinfo():
    logging.info("check")
    return jsonify(BUILDINFO)


@app.route("/status", methods=['GET'])
def get_request_status():
    possible_args = {"model_version_id", "request_id"}
    if set(request.args.keys()) != possible_args:
        return jsonify({"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400

    model_version_id = int(request.args['model_version_id'])
    explained_request_id = request.args['request_id']

    try:
        model = Model.find_by_id(hs_cluster, model_id=model_version_id)
    except ValueError as e:
        return jsonify({"message": f"Unable to found model version: {model_version_id}", "error": str(e)}), 404

    model_has_training_data = len(requests.get(f"{MONITORING_URL}/training_data?modelVersionId={model_version_id}").json()) > 0

    # TODO check for name of explained tensor name!
    method_support_statuses = utils.get_method_support_statuses(model.contract.predict, model_has_training_data)

    supported_methods = [method_status['name'] for method_status in method_support_statuses if method_status['supported']]

    try:
        rootcause_methods_statuses = []
        for method in supported_methods:
            response = {"method": method}
            instance_doc = db[method].find_one({"explained_request_id": {"$eq": explained_request_id},
                                                "model_version_id": {"$eq": model_version_id}})

            if instance_doc is None:
                response['status'] = {"state": "NOT_QUEUED"}
            else:
                task_id = instance_doc['celery_task_id']
                response['status'] = get_task_status(method=method, task_id=task_id).get_json()
            rootcause_methods_statuses.append(response)
    except Exception as e:
        logging.exception("Exception raised during fetching rootcause statuses")
        return jsonify({"message": f"{str(e)}"}), 500

    return jsonify(rootcause_methods_statuses)


@app.route("/supported_methods", methods=['GET'])
def get_supported_methods():
    possible_args = {"model_version_id"}
    if set(request.args.keys()) != possible_args:
        return jsonify({"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400

    model_version_id = int(request.args['model_version_id'])
    model = Model.find_by_id(hs_cluster, model_version_id)
    model_has_training_data = len(requests.get(f"{MONITORING_URL}/training_data?modelVersionId={model_version_id}").json()) > 0
    # TODO check for name of explained tensor name!
    supported_endpoints = utils.get_method_support_statuses(model.contract.predict, model_has_training_data)

    return jsonify({"method_support_statuses": supported_endpoints})


@app.route('/task_status', methods=["GET"])
def get_task_status():
    possible_args = {"method", "task_id"}
    if set(request.args.keys()) != possible_args:
        return jsonify({"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400

    method = request.args['method']
    task_id = request.args['task_id']

    available_methods = ['rise', 'anchor']
    if method == "rise":
        task = rise_tasks.tasks.rise_task.AsyncResult(task_id)
    elif method == "anchor":
        task = anchor_tasks.tasks.anchor_task.AsyncResult(task_id)
    else:
        raise ValueError("Invalid method - expected [{}], got".format(available_methods, method))

    response = {
        'state': task.state,
        'task_id': task_id
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


@app.route('/explanation', methods=["GET"])
def fetch_result():
    possible_args = {"model_version_id", "explained_request_id", "method"}
    if set(request.args.keys()) != possible_args:
        return jsonify({"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400

    method = request.args['method']
    request_id = request.args['explained_request_id']
    model_version_id = int(request.args['model_version_id'])

    explanation = db[method].find_one({"model_version_id": model_version_id, "explained_request_id": request_id})
    logging.info(explanation)

    if explanation:
        del explanation['_id']
        return jsonify(explanation)
    else:
        return Response(status=404)


@app.route("/explanation", methods=["POST"])
def launch_explanation_calculation():
    inp_json = request.get_json()

    if not validator.is_valid(inp_json):
        error_message = "\n".join([e.message for e in validator.iter_errors(inp_json)])
        return jsonify({"message": error_message}), 400

    model_version_id = inp_json['model_version_id']
    explained_request_id = inp_json['explained_request_id']
    method = inp_json['method']

    if method == "anchor":
        task = anchor_tasks.tasks.anchor_task
    elif method == "rise":
        task = rise_tasks.tasks.rise_task
    else:
        return 400, f"Invalid method: '{method}'"

    logging.info(f"Received request to explain request={explained_request_id} of model {model_version_id} with {method}")

    explanation_job_description = {'created_at': datetime.datetime.now(),
                                   'model_version_id': model_version_id,
                                   'explained_request_id': explained_request_id,
                                   'method': method}

    explanation_id = db[method].insert_one(explanation_job_description).inserted_id

    logging.debug("stored into db")

    task_async_result = task.delay(str(explanation_id))

    logging.debug("explanation task launched")

    explanation_id = objectid.ObjectId(explanation_id)
    db[method].find_one_and_update({"_id": explanation_id},
                                   {"$set": {"celery_task_id": task_async_result.id}})

    return jsonify({}), 202, {'Location': f'/rootcause/get_task_status?method={method}&task_id={task_async_result.id}'}


@app.route("/config", methods=['GET', 'PATCH'])
def get_params():
    possible_args = {"model_version_id", "method"}
    if set(request.args.keys()) != possible_args:
        return jsonify({"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400

    method = request.args['method']
    model_version_id = int(request.args['model_version_id'])

    # Fetch the latest config from the config collection in mongo
    config_doc = list(db[CONFIG_COLLECTION].find({"method": method,
                                                  "model_version_id": model_version_id}).sort([['_id', -1]]).limit(1))

    if not config_doc:
        current_config = utils.get_default_config(method)
    else:
        current_config = config_doc[0]['config']

    if request.method == 'GET':
        return jsonify(current_config)
    elif request.method == "PATCH":
        inp_json = request.get_json()

        # Patch current config with new values from inp_json
        new_config = {**current_config, **inp_json['config']}

        # Save patched version of config
        db[CONFIG_COLLECTION].insert_one({"method": method,
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
