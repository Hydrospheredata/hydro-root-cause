import datetime
import logging as logger
from time import sleep
from timeit import default_timer as timer
from typing import Callable

import grpc
import hydro_serving_grpc as hs
import hydro_serving_grpc.gateway as hsg
import numpy as np
import pandas as pd
import requests
from anchor2 import TabularExplainer
from bson import objectid
from hydrosdk.model import Model
from hydrosdk.servable import Servable
from pymongo import MongoClient
from pymongo.database import Database

import utils
from app import celery, get_mongo_client, hs_cluster, MONITORING_URL, ExplanationState
from utils import TabularContractType

BEAM_SELECTOR_PARAMETER_NAMES = ("delta", 'tolerance', 'batch_size', 'beam_size', "anchor_pool_size")


def get_anchor_classifier_fn(servable: Servable, feature_order, explained_tensor_name) -> Callable:
    # channel = grpc.secure_channel(SERVING_URL, credentials=grpc.ssl_channel_credentials())

    # TODO change before production deploy
    channel = grpc.secure_channel("hydro-serving.dev.hydrosphere.io", credentials=grpc.ssl_channel_credentials())
    # channel = grpc.insecure_channel("localhost:9090")
    stub = hsg.GatewayServiceStub(channel)

    def classifier_fn(x: np.array):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        output_aggregator = []

        for row in x:
            _x_df = pd.DataFrame(row.reshape((1, -1)), columns=feature_order).astype(int)

            # FIXME[MVP] use correct input dtypes
            # _x_df = _x_df.astype(contract.input_dtypes, copy=False)

            _x_proto = dict([(name, hs.TensorProto(dtype=hs.DT_INT64,
                                                   int64_val=value.to_list(),
                                                   tensor_shape=hs.TensorShapeProto())) for name, value in _x_df.iteritems()])

            # predict_request = hsg.ServablePredictRequest(servable_name=servable.name, data=_x_proto)

            predict_request = hsg.ServablePredictRequest(servable_name=servable.name, data=_x_proto)

            response = stub.ShadowlessPredictServable(predict_request)
            decoded_response = dict([(tensor_name, tensor_proto.int64_val) for tensor_name, tensor_proto in response.outputs.items()])[
                explained_tensor_name]
            output_aggregator.append([decoded_response])
        return np.array(output_aggregator)

    return classifier_fn


def get_anchor_explained_instance(request_id, feature_names):
    request_with_checks = requests.get(f"{MONITORING_URL}/checks/{request_id}").json()
    logger.info(f"{MONITORING_URL}/checks/{request_id}")
    request_wo_checks = dict([(f_name, request_with_checks[f_name]) for f_name in feature_names])
    return np.array(pd.Series(request_wo_checks))


def get_anchor_feature_names(reqstore_data: pd.DataFrame, model_tabular_contract_type):
    if model_tabular_contract_type == TabularContractType.SINGLE_TENSOR:
        return list(range(reqstore_data.shape[1]))
    elif model_tabular_contract_type == TabularContractType.SCALAR:
        return list(reqstore_data.columns)
    elif model_tabular_contract_type == TabularContractType.COLUMNAR:
        return list(reqstore_data.columns)
    else:
        raise ValueError("Unrecognized tabular signature type")


@celery.task
def anchor_task(explanation_id: str):
    #  TODO[Optimisation] if mongo anchor cache is we can check anchor presence by binary mask
    #  ex: db.collection.save({ _id: 1, a: 54, binaryValueofA: "00110110" })
    #  ex: db.collection.find( { a: { $bitsAllSet: [ 1, 5 ] } } )

    logger.info(f"Celery task for explanation {explanation_id} is launched.")

    mongo_client: MongoClient = get_mongo_client()
    db: Database = mongo_client['root_cause']

    def error_state(error_msg):
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

    # Fetch config for this model version. If no config provided - get default one
    job_config = db.configs.find_one({"method": "anchor",
                                      "model_version_id": model_version_id})
    if not job_config:
        job_config = utils.get_default_config("anchor")

    model_version = Model.find_by_id(hs_cluster, model_version_id)
    input_field_names = [t.name for t in model_version.contract.predict.inputs]
    output_field_names = [t.name for t in model_version.contract.predict.outputs]
    logger.info(f"{explanation_id} - Feature names used for calculating explanation: {input_field_names}")

    try:
        explained_tensor_name = job_config['output_explained_tensor_name']
        if explained_tensor_name not in output_field_names:
            raise ValueError(f"RootCause is configure to explain '{explained_tensor_name}' tensor. "
                             f"{model_version.name}v{model_version.version} have to return '{explained_tensor_name}' tensor.")

        explained_request: np.array = get_anchor_explained_instance(explained_request_id, input_field_names)

        logger.debug(f"{explanation_id} - restored request is: {explained_request}")
        assert explained_request.size > 0, "Restored sample should not be empty"
    except:
        error_state("Unable to load explained request")
        return str(explanation_id)

    try:
        logger.info(f"{explanation_id} - connecting to monitoring to get url to training data")
        s3_training_data_path = requests.get(f"{MONITORING_URL}/training_data?modelVersionId={model_version_id}").json()[0]
        # TODO change to Monitoring get
        s3_training_data_path = "s3://feature-lake/training-data/53/training_data532034320489197967253.csv"

        logger.info(f"{explanation_id} - started fetching training data, url: {s3_training_data_path}")
        started_downloading_data_time = timer()
        training_data = pd.read_csv(s3_training_data_path)
        logger.info(f"{explanation_id} - finished loading training data in {timer() - started_downloading_data_time:.2f} seconds")

        training_data = training_data.sample(job_config['training_subsample_size'])
        training_data = training_data[input_field_names]  # Reorder columns according to their order
    except:
        error_state("Unable to load training data")
        return str(explanation_id)

    try:
        # Create temporary servable, so main servable won't be affected
        tmp_servable: Servable = Servable.create(hs_cluster,
                                                 model_version.name,
                                                 model_version.version,
                                                 metadata={"created_by": "rootcause"})
        # TODO poll for service status!
        sleep(10)
    except:
        error_state("Unable to create a new servable")
        return str(explanation_id)

    try:
        anchor_explainer = TabularExplainer()
        anchor_explainer.fit(data=training_data,
                             ordinal_features_idx=job_config['ordinal_features_idx'],
                             label_decoders=job_config['label_decoders'],
                             oh_encoded_categories=job_config['oh_encoded_categories'])

        logger.info(f"{explanation_id} - servable {tmp_servable.name} is ready.")

        beam_selector_parameters = dict([(k, v) for k, v in job_config.items() if k in BEAM_SELECTOR_PARAMETER_NAMES])

        explanation = anchor_explainer.explain(explained_request,
                                               classifier_fn=get_anchor_classifier_fn(servable=tmp_servable,
                                                                                      feature_order=list(training_data.columns),
                                                                                      explained_tensor_name=explained_tensor_name),
                                               threshold=job_config['threshold'],
                                               selector_params=beam_selector_parameters)
    except:
        error_state("Error happened during iterating over possible explanations")
        return str(explanation_id)
    finally:
        # Remove temporary servable
        Servable.delete(hs_cluster, tmp_servable.name)

    result_json = {"explanation": [str(p) for p in explanation.predicates],
                   "coverage": explanation.coverage(),
                   "precision": explanation.precision()}

    db.anchor.update_one({"_id": explanation_id}, {"$set": {'result': result_json,
                                                            "state": ExplanationState.SUCCESS.name,
                                                            "description": "Explanation successfully computed",
                                                            "completed_at": datetime.datetime.now()}})

    logger.info(f"{explanation_id} - Finished computing explanation.")

    return str(explanation_id)
