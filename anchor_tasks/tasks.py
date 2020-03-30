import datetime
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
from loguru import logger
from pymongo import MongoClient
from pymongo.database import Database

import utils
from app import celery, get_mongo_client, hs_cluster, MONITORING_URL
from utils import TabularContractType


def get_anchor_classifier_fn(servable: Servable, feature_order, explained_tensor_name) -> Callable:
    # channel = grpc.secure_channel(SERVING_URL, credentials=grpc.ssl_channel_credentials())

    channel = grpc.insecure_channel("localhost:9090")
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

            predict_request = hsg.ServablePredictRequest(servable_name="adult-classification-1-bold-wildflower", data=_x_proto)

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
    explanation_id = objectid.ObjectId(explanation_id)
    job_json = db.anchor.find_one({"_id": explanation_id})

    if 'result' in job_json:
        # TODO check it before starting celery task!
        return str(explanation_id)

    job_json = db.anchor.find_one_and_update({"_id": explanation_id},
                                             {"$set": {"started_at": datetime.datetime.now(),
                                                       "status": "STARTED"}})

    model_version_id = job_json['model_version_id']
    explained_request_id = job_json['explained_request_id']

    # Fetch config for this model version. If no config provided - get default one
    job_config = db.configs.find_one({"method": "anchor",
                                      "model_version_id": model_version_id})
    if not job_config:
        job_config = utils.get_default_config("anchor")

    model_version = Model.find_by_id(hs_cluster, model_version_id)
    input_field_names = [t.name for t in model_version.contract.predict.inputs]
    output_field_names = [t.name for t in model_version.contract.predict.outputs]
    logger.info(f"Explanation {explanation_id}. Feature names used for calculating explanation: {input_field_names}")

    explained_tensor_name = job_config['output_explained_tensor_name']
    if explained_tensor_name not in output_field_names:
        raise ValueError(f"RootCause is configure to explain '{explained_tensor_name}' tensor. "
                         f"{model_version.name}v{model_version.version} have to return '{explained_tensor_name}' tensor.")

    explained_request: np.array = get_anchor_explained_instance(explained_request_id, input_field_names)

    logger.debug(f"Explanation {explanation_id}: restored request is: {explained_request}")
    assert explained_request.size > 0, "Restored sample should not be empty"

    logger.info(f"Explanation {explanation_id}: connecting to monitoring to get url to training data")
    s3_training_data_path = requests.get(f"{MONITORING_URL}/training_data?modelVersionId={model_version_id}").json()[0]
    logger.info(f"Explanation {explanation_id}: started fetching training data, url: {s3_training_data_path}")

    # TODO remove after deployement
    s3_training_data_path = "s3://feature-lake/training-data/53/training_data532034320489197967253.csv"

    training_data = pd.read_csv(s3_training_data_path)
    logger.info(f"Explanation {explanation_id}: finished loading training data.")

    training_data = training_data.sample(job_config['training_subsample_size'])
    training_data = training_data[input_field_names]

    anchor_explainer = TabularExplainer()
    anchor_explainer.fit(data=training_data,
                         ordinal_features_idx=job_config['ordinal_features_idx'],
                         label_decoders=job_config['label_encoders'],
                         oh_encoded_categories=job_config['oh_encoded_categories'])

    # Create temporary servable, so main servable won't be affected
    tmp_servable: Servable = Servable.create(hs_cluster,
                                             model_version.name,
                                             model_version.version,
                                             metadata={"private": "true",
                                                       "created_by": "rootcause"})

    # FIXME Servable create is async right now, I have no proof that I can use servable to send requests to it
    logger.info(f"explanation {explanation_id}: servable {tmp_servable.name} is ready.")

    BEAM_SELECTOR_PARAMETER_NAMES = ("delta", 'tolerance', 'batch_size', 'beam_size', "anchor_pool_size")
    beam_selector_parameters = dict([(k, v) for k, v in job_config.items() if k in BEAM_SELECTOR_PARAMETER_NAMES])

    explanation = anchor_explainer.explain(explained_request,
                                           classifier_fn=get_anchor_classifier_fn(servable=tmp_servable,
                                                                                  feature_order=list(training_data.columns),
                                                                                  explained_tensor_name=explained_tensor_name),
                                           threshold=job_config['threshold'],
                                           selector_params=beam_selector_parameters)

    # TODO[Optimisation] Check if there are any more explanations in queue for this model version (?) If yes - do not remove servable.
    # Remove temporary servable
    Servable.delete(hs_cluster, tmp_servable.name)

    result_json = {"explanation": [str(p) for p in explanation.predicates],
                   "coverage": explanation.coverage(),
                   "precision": explanation.precision()}

    db.anchor.update_one({"_id": explanation_id}, {"$set": {'result': result_json,
                                                            "completed_at": datetime.datetime.now()}})

    logger.info(f"Explanation {explanation_id}. Finished computing explanation.")

    return str(explanation_id)
