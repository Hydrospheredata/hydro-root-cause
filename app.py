import datetime
import os

from bson import objectid
from celery import Celery
from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
from hydro_serving_grpc.reqstore import reqstore_client
from loguru import logger
from pymongo import MongoClient

import utils
from client import HydroServingClient

REQSTORE_URL = os.getenv("REQSTORE_URL", "managerui:9090")
SERVING_URL = os.getenv("SERVING_URL", "managerui:9090")

MONGO_URL = os.getenv("MONGO_URL", "mongodb")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")


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

import anchor_tasks
import rise_tasks


@app.route("/")
def hello():
    return "hydro_root_cause_service"


@app.route("/supported_methods", methods=['GET'])
def get_supported_methods():
    # TODO add jsonschema validation
    model_json = request.get_json()
    model_name = model_json['model']['name']
    model_version = model_json['model']['version']
    model = hs_client.get_model(model_name, model_version)
    supported_endpoints = utils.get_supported_endpoints(model.contract)
    return jsonify({"supported_endpoints": supported_endpoints})


@app.route('/status/<method>/<task_id>', methods=["GET"])
def task_status(task_id, method):
    if method == "rise":
        task = rise_tasks.tasks.rise_task.AsyncResult(task_id)
    elif method == "anchor":
        task = anchor_tasks.tasks.anchor_task.AsyncResult(task_id)
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

    return jsonify(explanation)


@app.route("/anchor", methods=['POST'])
def anchor():
    # TODO add jsonschema validation

    inp_json = request.get_json()
    logger.info(f"Received request to explain {str(inp_json['model'])} with anchor")

    inp_json['created_at'] = datetime.datetime.now()

    anchor_explanation_id = db.anchor_explanations.insert_one(inp_json).inserted_id
    task = anchor_tasks.tasks.anchor_task.delay(str(anchor_explanation_id))

    return jsonify({}), 202, {'Location': '/rootcause' + url_for('task_status', task_id=task.id, _external=False, method="anchor")}


@app.route("/rise", methods=['POST'])
def rise():
    # TODO add jsonschema validation

    inp_json = request.get_json()
    inp_json['created_at'] = datetime.datetime.now()

    logger.info(f"Received request to explain {str(inp_json['model'])} with rise")

    rise_explanation_id = db.rise_explanations.insert_one(inp_json).inserted_id
    task = rise_tasks.tasks.rise_task.delay(str(rise_explanation_id))

    return jsonify({}), 202, {'Location': '/rootcause' + url_for('task_status', _external=False, task_id=task.id, method="rise")}


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
