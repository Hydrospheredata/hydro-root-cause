import datetime
import json
import logging as logger
from timeit import default_timer as timer
from typing import Callable

import numpy as np
import pandas as pd
import requests
import sseclient
from anchor2 import TabularExplainer
from bson import objectid
from hydrosdk.cluster import Cluster
from hydrosdk.modelversion import ModelVersion
from hydrosdk.servable import Servable, ServableStatus
from hydrosdk.contract import ProfilingType
from pymongo import MongoClient
from pymongo.database import Database
from s3fs.core import S3FileSystem

import utils
from app import celery, get_mongo_client, MONITORING_URL, ExplanationState, S3_ENDPOINT, HS_CLUSTER_ADDRESS, \
    GRPC_ADDRESS

BEAM_SELECTOR_PARAMETER_NAMES = ("delta", 'tolerance', 'batch_size', 'beam_size', "anchor_pool_size")


def get_anchor_classifier_fn(servable: Servable, feature_order, explained_tensor_name) -> Callable:
    predictor = servable.predictor(monitorable=False)

    def classifier_fn(x: np.array):
        x = x.astype("int")
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        output_aggregator = []
        for row in x:
            request = {feature_name: feature_value for feature_name, feature_value in zip(feature_order, row)}
            response = predictor.predict(request)
            output_aggregator.append(response[explained_tensor_name])
        return np.array(output_aggregator).reshape(-1, 1)

    return classifier_fn


def get_anchor_explained_instance(request_id, feature_names):
    request_with_checks = requests.get(f"{MONITORING_URL}/checks/{request_id}").json()
    logger.info(f"{MONITORING_URL}/checks/{request_id}")
    request_wo_checks = dict([(f_name, request_with_checks[f_name]) for f_name in feature_names])
    return np.array(pd.Series(request_wo_checks))


@celery.task
def anchor_task(explanation_id: str):
    #  [Optimisation?] if mongo anchor cache is we can check anchor presence by binary mask
    #  ex: db.collection.save({ _id: 1, a: 54, binaryValueofA: "00110110" })
    #  ex: db.collection.find( { a: { $bitsAllSet: [ 1, 5 ] } } )

    logger.info(f"Celery task for explanation {explanation_id} is launched.")

    try:
        mongo_client: MongoClient = get_mongo_client()
        db: Database = mongo_client['root_cause']
    except:
        logger.error("Failed to connect to Mongodb")
        return explanation_id

    def log_error_state(error_msg):
        logger.error(error_msg)
        db.anchor.find_one_and_update({"_id": objectid.ObjectId(explanation_id)},
                                      {"$set": {"state": ExplanationState.FAILED.name,
                                                "description": error_msg}})

    job_json = db.anchor.find_one_and_update({"_id": objectid.ObjectId(explanation_id)},
                                             {"$set": {"started_at": datetime.datetime.now(),
                                                       "state": ExplanationState.STARTED.name,
                                                       "description": "Explanation is being calculated right now."}})

    model_version_id = int(job_json['model_version_id'])
    explained_request_id = job_json['explained_request_id']
    logger.info(job_json)

    try:
        job_config = utils.get_latest_config(db, "anchor", model_version_id)
    except Exception as e:
        log_error_state(f"Failed to load config for this job {e}")
        return str(explanation_id)

    try:
        hs_cluster = Cluster(HS_CLUSTER_ADDRESS, grpc_address=GRPC_ADDRESS)
        model_version = ModelVersion.find_by_id(hs_cluster, model_version_id)
        input_field_names = [t.name for t in model_version.contract.predict.inputs]
        output_field_names = [t.name for t in model_version.contract.predict.outputs]
        ordinal_features_idx = [idx for idx, inpt in enumerate(model_version.contract.predict.inputs) if ProfilingType(inpt.profile) == ProfilingType.ORDINAL]
        logger.info(f"{explanation_id} - Feature names used for calculating explanation: {input_field_names}.")
    except Exception as e:
        log_error_state(f"Failed to connect to the model. {e}")
        return str(explanation_id)

    try:
        explained_tensor_name = job_json['explained_output_field_name']  # Fix here
        if explained_tensor_name not in output_field_names:
            raise ValueError(f"RootCause is configure to explain '{explained_tensor_name}' tensor. "
                             f"{model_version.name}v{model_version.version} have to return '{explained_tensor_name}' tensor.")

        explained_request: np.array = get_anchor_explained_instance(explained_request_id, input_field_names)

        logger.debug(f"{explanation_id} - restored request is: {explained_request}")
        if explained_request.size < 0:
            raise ValueError("Explained request should not be empty")
    except Exception as e:
        log_error_state(f"Unable to load explained request. {e}.")
        return str(explanation_id)

    try:
        logger.info(f"{explanation_id} - connecting to monitoring to get url to training data")
        s3_training_data_path = \
        requests.get(f"{MONITORING_URL}/training_data?modelVersionId={model_version_id}").json()[0]

        logger.info(f"{explanation_id} - started fetching training data, url: {s3_training_data_path}")
        started_downloading_data_time = timer()

        if S3_ENDPOINT:
            s3 = S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT})
            training_data = pd.read_csv(s3.open(s3_training_data_path, mode='rb'))
        else:
            training_data = pd.read_csv(s3_training_data_path)

        logger.info(
            f"{explanation_id} - finished loading training data in {timer() - started_downloading_data_time:.2f} seconds")

        training_data = training_data.sample(job_config['training_subsample_size'])
        training_data = training_data[input_field_names]  # Reorder columns according to their order
    except Exception as e:
        log_error_state(f"Unable to load training data. {e}")
        return str(explanation_id)

    try:

        tmp_servable: Servable = Servable.create(hs_cluster, model_name=model_version.name, version=model_version.version,
                                       metadata={"created_by": "rootcause"})

        utils.servable_lock_till_serving(hs_cluster, tmp_servable.name)
        tmp_servable: Servable = Servable.find_by_name(hs_cluster, tmp_servable.name)


    except Exception as e:
        log_error_state(f"Unable to create a new servable. {e}")
        return str(explanation_id)

    try:
        anchor_explainer = TabularExplainer()
        anchor_explainer.fit(data=training_data,
                             ordinal_features_idx=ordinal_features_idx,
                             label_decoders=job_config['label_decoders'],
                             oh_encoded_categories=job_config['oh_encoded_categories'])

        logger.info(f"{explanation_id} - servable {tmp_servable.name} is ready.")

        beam_selector_parameters = dict([(k, v) for k, v in job_config.items() if k in BEAM_SELECTOR_PARAMETER_NAMES])

        explanation, explained_field_value = \
            anchor_explainer.explain(x=explained_request,
                                     classifier_fn=get_anchor_classifier_fn(servable=tmp_servable,
                                                                            feature_order=list(training_data.columns),
                                                                            explained_tensor_name=explained_tensor_name),
                                     threshold=job_config['threshold'],
                                     selector_params=beam_selector_parameters)
    except Exception as e:
        log_error_state(f"Error happened during iterating over possible explanations. {e}")
        return str(explanation_id)
    finally:
        # Remove temporary servable
        Servable.delete(hs_cluster, tmp_servable.name)

    try:
        result_json = {"explanation": [str(p) for p in explanation.predicates],
                       "coverage": explanation.coverage(),
                       "precision": explanation.precision(),
                       "explained_field_name": str(explained_tensor_name),
                       "explained_field_value": int(explained_field_value)}

        db.anchor.update_one({"_id": objectid.ObjectId(explanation_id)},
                             {"$set": {'result': result_json,
                                       "state": ExplanationState.SUCCESS.name,
                                       "description": "Explanation successfully computed",
                                       "completed_at": datetime.datetime.now()}})

        logger.info(f"{explanation_id} - Finished saving computed explanation.")
    except Exception as e:
        log_error_state(f"Error happened during saving explanation. {e}")
        return str(explanation_id)

    return str(explanation_id)


