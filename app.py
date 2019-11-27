import datetime
import json
import os
import sys
from subprocess import check_output

from bson import objectid
from celery import Celery
from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
from hydro_serving_grpc.reqstore import reqstore_client
from jsonschema import Draft7Validator
from loguru import logger
from pymongo import MongoClient
from waitress import serve

import utils
from client import HydroServingClient

DEBUG_ENV = bool(os.getenv("DEBUG_ENV", True))
VERSION = "$Id$"

REQSTORE_URL = os.getenv("REQSTORE_URL", "managerui:9090")
SERVING_URL = os.getenv("SERVING_URL", "managerui:9090")

MONGO_URL = os.getenv("MONGO_URL", "mongodb")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")

with open("./root_cause_request_json_schema.json") as f:
    REQUEST_JSON_SCHEMA = json.load(f)
    validator = Draft7Validator(REQUEST_JSON_SCHEMA)


def get_mongo_client():
    return MongoClient(host=MONGO_URL, port=MONGO_PORT, maxPoolSize=200,
                       username=MONGO_USER, password=MONGO_PASS,
                       authSource=MONGO_AUTH_DB)


mongo_client = get_mongo_client()

db = mongo_client['root_cause']

hs_client = HydroServingClient(SERVING_URL)
rs_client = reqstore_client.ReqstoreClient(REQSTORE_URL, insecure=True)

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

import anchor_tasks
import rise_tasks


@app.route("/", methods=['GET'])
def hello():
    return "Hi! I am RootCause Service"


@app.route("/buildinfo", method=['GET'])
def buildinfo():
    branch_name = check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf8").strip()
    py_version = sys.version
    return jsonify({"version": VERSION,
                    "name": "root-cause",
                    "pythonVersion": py_version,
                    "gitCurrentBranch": branch_name,
                    "gitHeadCommit": VERSION})


@app.route("/status", methods=['GET'])
def get_instance_status():
    possible_args = {"model_name", "model_version", "uid", "ts"}
    if set(request.args.keys()) != possible_args:
        return jsonify({"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400

    try:
        model_name = request.args.get('model_name')
        model_version = int(request.args.get('model_version'))
        uid = int(request.args.get('uid'))
        timestamp = int(request.args.get('ts'))
    except Exception as e:
        return jsonify({"message": f"Was unable to cast one of {('model_version', 'uid', 'ts')} to int"}), 400

    try:
        model = hs_client.get_model(model_name, model_version)
    except ValueError as e:
        return jsonify({"message": f"Unable to found {model_name}v{model_version}"}), 404

    supported_endpoints = utils.get_supported_endpoints(model.contract)
    logger.debug(f"Supported endpoints {supported_endpoints}")

    try:
        rootcause_methods_statuses = []
        for method in supported_endpoints:
            response = {"method": method}
            instance_doc = db[method].find_one({"explained_instance.uid": {"$eq": uid},
                                                "explained_instance.timestamp": {"$eq": timestamp},
                                                "model.name": {"$eq": model_name},
                                                "model.version": {"$eq": model_version}})
            if instance_doc is None:
                response['status'] = {"state": "NOT_QUEUED"}
            else:
                task_id = instance_doc['celery_task_id']
                response['status'] = get_task_status(method=method, task_id=task_id).get_json()
            rootcause_methods_statuses.append(response)
    except Exception as e:
        return jsonify({"message": f"{str(e)}"}), 500

    return jsonify(rootcause_methods_statuses)


@app.route("/supported_methods", methods=['GET'])
def get_supported_methods():
    req_json = request.get_json()
    model_name = req_json['model']['name']
    model_version = req_json['model']['version']
    model = hs_client.get_model(model_name, model_version)
    supported_endpoints = utils.get_supported_endpoints(model.contract)
    return jsonify({"supported_methods": supported_endpoints})


@app.route('/task_status/<method>/<task_id>', methods=["GET"])
def get_task_status(method, task_id):
    avaiable_methods = ['rise', 'anchor']
    if method == "rise":
        task = rise_tasks.tasks.rise_task.AsyncResult(task_id)
    elif method == "anchor":
        task = anchor_tasks.tasks.anchor_task.AsyncResult(task_id)
    else:
        raise ValueError("Invalid method - expected [{}], got".format(avaiable_methods, method))

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


@app.route('/fetch_result/<method>/<result_id>', methods=["GET"])
def fetch_result(result_id, method):
    explanation = db[method].find_one({"_id": objectid.ObjectId(result_id)})
    del explanation['_id']

    return jsonify(explanation)


@app.route("/anchor", methods=['POST'])
def anchor():
    inp_json = request.get_json()
    if not validator.is_valid(inp_json):
        error_message = "\n".join(validator.iter_errors(inp_json))
        return jsonify({"message": error_message}), 400

    logger.info(f"Received request to explain {str(inp_json['model'])} with anchor")

    inp_json['created_at'] = datetime.datetime.now()

    anchor_explanation_id = db.anchor.insert_one(inp_json).inserted_id
    task = anchor_tasks.tasks.anchor_task.delay(str(anchor_explanation_id))

    explanation_id = objectid.ObjectId(anchor_explanation_id)
    db.anchor.find_one_and_update({"_id": explanation_id},
                                  {"$set": {"celery_task_id": task.id}})

    return jsonify({}), 202, {'Location': '/rootcause' + url_for('get_task_status', task_id=task.id, _external=False, method="anchor")}


@app.route("/rise", methods=['POST'])
def rise():
    inp_json = request.get_json()
    if not validator.is_valid(inp_json):
        error_message = "\n".join(validator.iter_errors(inp_json))
        return jsonify({"message": error_message}), 400

    inp_json['created_at'] = datetime.datetime.now()

    logger.info(f"Received request to explain {str(inp_json['model'])} with rise")

    rise_explanation_id = db.rise.insert_one(inp_json).inserted_id
    task = rise_tasks.tasks.rise_task.delay(str(rise_explanation_id))

    explanation_id = objectid.ObjectId(rise_explanation_id)
    db.rise.find_one_and_update({"_id": explanation_id},
                                {"$set": {"celery_task_id": task.id}})

    return jsonify({}), 202, {'Location': '/rootcause' + url_for('get_task_status', _external=False, task_id=task.id, method="rise")}


if __name__ == "__main__":
    if not DEBUG_ENV:
        serve(app, host='0.0.0.0', port=5000)
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)
