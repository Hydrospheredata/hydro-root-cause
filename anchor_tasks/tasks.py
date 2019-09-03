import datetime
from typing import Dict, List, Callable

import numpy as np
import pandas as pd
from anchor2 import TabularExplainer
from bson import objectid
from loguru import logger
from pymongo import MongoClient
from pymongo.database import Database

import utils
from app import celery, rs_client, hs_client, get_mongo_client
from client import HydroServingServable
from contract import HSContract
from utils import TabularContractType


def get_anchor_classifier_fn(servable: HydroServingServable, contract_type: TabularContractType, feature_order) -> Callable:
    contract: HSContract = servable.contract
    if contract_type == TabularContractType.COLUMNAR:
        def classifier_fn(x: np.array):
            x_df = pd.DataFrame(x, columns=feature_order)
            x_df = x_df.astype(contract.input_dtypes, copy=False)  # Anchor permutator can change int -> float, so we need to cast them back
            return servable(x_df)['classes']
    elif contract_type == TabularContractType.SCALAR:
        def classifier_fn(x: np.array):
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            output_aggregator = []
            for row in x:
                _x_df = pd.DataFrame(row.reshape((1, -1)), columns=feature_order)
                _x_df = _x_df.astype(contract.input_dtypes, copy=False)
                output_aggregator.append(servable(_x_df)['classes'])
            return np.array(output_aggregator)
    elif contract_type == TabularContractType.SINGLE_TENSOR:
        def classifier_fn(x: np.array):
            input_dtype = list(contract.input_dtypes.values())[0]
            x = x.astype(input_dtype)  # Anchor permutator can change int -> float, so we need to cast them back
            return servable(x)['classes']
    else:
        raise ValueError("Invalid contract. Impossible to construct proper classification_fn")
    return classifier_fn


def get_anchor_subsample(_rs_client, model, contract_type: TabularContractType, subsample_size=5000) -> pd.DataFrame:
    rs_entries = utils.fetch_reqstore_entries(_rs_client, model, subsample_size)

    if contract_type == TabularContractType.COLUMNAR:
        subsample = utils.extract_subsample_from_columnar_reqstore_entries(rs_entries, model)
    elif contract_type == TabularContractType.SCALAR:
        subsample = utils.extract_subsample_from_scalar_reqstore_entries(rs_entries, model)
    elif contract_type == TabularContractType.SINGLE_TENSOR:
        subsample = utils.extract_subsample_from_tensor_reqstore_entries(rs_entries, model)
    else:
        raise ValueError("Invalid contract. Impossible to get proper subsample")
    return subsample


def get_anchor_explained_instance(model,
                                  folder,
                                  ts,
                                  reqstore_uid,
                                  model_tabular_contract_type, feature_order):
    input_tensors = utils.get_reqstore_request(model.contract, rs_client, folder, ts, reqstore_uid)

    if model_tabular_contract_type == TabularContractType.SINGLE_TENSOR:
        input_array = input_tensors['input']
        if input_array.shape[0] != 1:
            raise ValueError("Request has to have a single sample")
        x: np.ndarray = input_array[0]
    elif model_tabular_contract_type == TabularContractType.SCALAR:
        x = np.array([input_tensors[name] for name in feature_order])
    elif model_tabular_contract_type == TabularContractType.COLUMNAR:
        input_tensors_flattened = dict([(k, v.flatten()) for k, v in input_tensors.items()])
        input_df = pd.DataFrame.from_dict(input_tensors_flattened)
        if input_df.shape[0] != 1:
            raise ValueError("Request has to have a single sample")
        x: np.ndarray = np.array(input_df.iloc[0][feature_order])
    else:
        raise ValueError("Unrecognized tabular contract type")
    return x


def get_anchor_feature_names(reqstore_data: pd.DataFrame, model_tabular_contract_type):
    if model_tabular_contract_type == TabularContractType.SINGLE_TENSOR:
        return list(range(reqstore_data.shape[1]))
    elif model_tabular_contract_type == TabularContractType.SCALAR:
        return list(reqstore_data.columns)
    elif model_tabular_contract_type == TabularContractType.COLUMNAR:
        return list(reqstore_data.columns)
    else:
        raise ValueError("Unrecognized tabular contract type")


@celery.task
def anchor_task(explanation_id: str):
    #  TODO if mongo anchor cache is we can check anchor presence by binary mask
    #  ex: db.collection.save({ _id: 1, a: 54, binaryValueofA: "00110110" })
    #  ex: db.collection.find( { a: { $bitsAllSet: [ 1, 5 ] } } )

    mongo_client: MongoClient = get_mongo_client()
    db: Database = mongo_client['root_cause']

    explanation_id = objectid.ObjectId(explanation_id)
    job_json = db.anchor.find_one_and_update({"_id": explanation_id},
                                             {"$set": {"started_at": datetime.datetime.now()}})

    if 'result' in job_json:
        return str(explanation_id)

    model_name = job_json['model']['name']
    model_version = job_json['model']['version']
    config = job_json.get('config', {})

    model = hs_client.get_model(model_name, model_version)
    model_tabular_contract_type = utils.get_tabular_contract_type(model.contract)

    ts = job_json['explained_instance']['timestamp']
    folder = str(model.id)
    reqstore_uid = job_json['explained_instance']['uid']

    logger.info(
        f"Initiated task to explain numerical sample folder:{folder}, ts:{ts}, "
        f"uid:{reqstore_uid} under model {model_name}_{model_version} with anchor")

    if 'classes' not in model.contract.output_names:
        raise ValueError("Model have to return 'classes' tensor of shape [-1; 1]")

    x = get_anchor_explained_instance(model, folder, ts, reqstore_uid, model_tabular_contract_type,
                                      feature_order=model.contract.input_names)

    logger.debug(f"Restored X to explain: {x}")
    assert x.size > 0, "Restored sample should not be empty"

    # Create temporary servable, so main servable won't be affected
    tmp_servable: HydroServingServable = hs_client.deploy_servable(model_name, model_version)
    logger.info("Servable deployed")

    # Convert config dicts to appropriate types
    # label_decoders: Dict[int, List[str]] = dict([(int(k), v) for (k, v) in config['label_decoders'].items()])
    label_decoders: Dict[int, List[str]] = dict()  # This is a temp workaround, since we can't pass this config through UI

    # oh_encoded_categories: Dict[str, List[int]] = dict(
    #     [(k, [int(v) for v in vs]) for (k, vs) in config['oh_encoded_categories'].items()])
    oh_encoded_categories: Dict[str, List[int]] = dict()  # This is a temp workaround, since we can't pass this config through UI

    anchor_explainer = TabularExplainer()

    # Get subsample to work with
    logger.info("Start fetching reqstore data")
    reqstore_data = get_anchor_subsample(rs_client, model, model_tabular_contract_type, config.get("subsample_size", 5000))

    logger.info("Finished fetching reqstore data.")

    feature_names = get_anchor_feature_names(reqstore_data, model_tabular_contract_type)
    logger.info("Feature names used in anchor: ", feature_names)

    anchor_explainer.fit(data=reqstore_data,
                         label_decoders=label_decoders,
                         ordinal_features_idx=config.get('ordinal_features_idx', [0, 11]),
                         oh_encoded_categories=oh_encoded_categories,
                         feature_names=feature_names)

    explanation = anchor_explainer.explain(x, classifier_fn=get_anchor_classifier_fn(tmp_servable, model_tabular_contract_type,
                                                                                     feature_order=feature_names))

    result_json = {"explanation": [str(p) for p in explanation.predicates],
                   "coverage": explanation.coverage(),
                   "precision": explanation.precision()}

    # Store explanation in MongoDB 'root_cause' db
    db.anchor.update_one({"_id": explanation_id}, {"$set": {'result': result_json,
                                                            "completed_at": datetime.datetime.now()}})

    logger.info(f"Finished task to explain {model_name}_{model_version} with anchor")

    # Remove temporary servable
    tmp_servable.delete()

    return str(explanation_id)
