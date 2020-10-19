import json
from typing import Tuple, Dict

import requests
import sseclient
from hydro_serving_grpc import DT_INT64, DT_INT32, DT_INT16, DT_INT8
from hydro_serving_grpc.contract import ModelSignature
from hydrosdk import Cluster
from hydrosdk.modelversion import ModelVersion
from hydrosdk.servable import Servable, ServableStatus

from app import hs_cluster, MONITORING_URL


class AlwaysTrueObj(object):
    def __eq__(self, other):
        return True


AnyDimSize = AlwaysTrueObj()


def contract_is_supported_by_rise(signature: ModelSignature, training_data_available,
                                  output_explained_tensor_name="probabilities") -> Tuple[bool, str]:
    # Rise supports batch and non\batch request of images of grayscale or 3 channeled.
    rise_supported_input_shapes = [(-1, AnyDimSize, AnyDimSize, 3),
                                   (-1, AnyDimSize, AnyDimSize, 1),
                                   (-1, AnyDimSize, AnyDimSize), ]

    rise_supported_probabilities_shapes = [(-1, AnyDimSize), ]

    if output_explained_tensor_name in [tensor.name for tensor in signature.outputs]:
        probabilities_field = next(filter(lambda t: t.name == output_explained_tensor_name, signature.outputs))
        probabilities_shape = tuple(map(lambda dim: dim.size, probabilities_field.shape.dim))
        if probabilities_shape not in rise_supported_probabilities_shapes:
            return False, f"Shape {probabilities_shape} of explained field '{output_explained_tensor_name}' is not supported" \
                          f" You can change explained tensor by calling config"

    else:
        return False, f"No output explained field named '{output_explained_tensor_name}' in contract signature was found" \
                      f" You can change explained tensor by calling config"

    if len(signature.inputs) != 1:
        # For now we support only signatures which take single image as an input
        return False, "Only signatures with single tensor is supported right now"
    else:
        input_tensor = signature.inputs[0]
        input_tensor_shape = tuple(map(lambda dim: dim.size, input_tensor.shape.dim))
        if input_tensor_shape not in rise_supported_input_shapes:
            return False, f"Input tensor shape {input_tensor_shape} is not supported"
        # FIXME check for type. supported - int, uint, double, float

    return True, "Everything is fine"


def check_model_version_support(method, model_version_id, explained_output_field_name) -> Tuple[bool, str]:
    """
    Checks whether model version is supported by particular method. Requires access to latest config to see which fields are being explained
    :param db:
    :param method:
    :param model_version_id:
    :param explained_output_field_name:
    :return:
    """
    if method == 'anchor':
        has_training_data = len(
            requests.get(f"{MONITORING_URL}/training_data?modelVersionId={model_version_id}").json()) > 0

        if not has_training_data:
            return False, "Unable to explain model version without training data."

        model_version = ModelVersion.find_by_id(hs_cluster, model_version_id)

        if model_version.is_external:
            return False, f"External Models are not supported"

        signature = model_version.contract.predict

        if explained_output_field_name in [tensor.name for tensor in signature.outputs]:
            classes_field = next(filter(lambda t: t.name == explained_output_field_name, signature.outputs))
            classes_field_shape = tuple(map(lambda dim: dim.size, classes_field.shape.dim))
            if classes_field_shape != tuple():
                return False, f"Explained tensor '{explained_output_field_name}' is of unsupported shape - {classes_field_shape}." \
                              f" You can change explained tensor by calling config"
        else:
            return False, f"Explained tensor is '{explained_output_field_name}' and it is not present in signature." \
                          f" You can change explained tensor by calling config"

        input_tensor_shapes = [tuple(map(lambda dim: dim.size, input_tensor.shape.dim)) for input_tensor in
                               signature.inputs]
        if not all([shape == tuple() for shape in input_tensor_shapes]):
            return False, "This signature has invalid type, only signatures with all scalar fields are supported right now"

        input_tensor_dtypes = [input_tensor.dtype for input_tensor in signature.inputs]
        if not all([dtype in {DT_INT64, DT_INT32, DT_INT16, DT_INT8} for dtype in input_tensor_dtypes]):
            return False, "This signature fields have invalid dtypes, only signatures with fields of int types are supported right now"

        return True, "Everything is fine"
    else:
        # TODO check for rise compatibility
        return False, "rise is unavailable for now"


def get_latest_config(db, method, model_version_id) -> Dict:
    """
    Fetch the latest config from the config collection in mongo, if no config is stored in mongo, use default one

    :param db: config_coll
    :param method:
    :param model_version_id:
    :return: dict with config variables
    """
    config_doc = list(db["config"].find({"method": method,
                                         "model_version_id": model_version_id}).sort([['_id', -1]]).limit(1))

    if not config_doc:
        latest_config = get_default_config(method)
    else:
        latest_config = config_doc[0]['config']
    return latest_config


def get_default_config(method):
    """
    Default configuration for each interpretability method
    :param method:
    :return: dictionary with default parameters
    """
    if method == "rise":
        return {"number_of_masks": 100,
                "mask_granularity": 100,
                "mask_density": 100, }
    elif method == "anchor":

        return {"training_subsample_size": 10000,
                # "precision_threshold": 0.5,
                "label_decoders": {},
                "threshold": 0.9,
                "oh_encoded_categories": {},
                "anchor_pool_size": 10,
                "beam_size": 5,
                "batch_size": 100,
                "tolerance": 0.35,
                "delta": 0.25}
    else:
        raise ValueError(f"Invalid method {method} passed")

def servable_lock_till_serving(cluster: Cluster, servable_name: str, timeout_messages: int = 30) -> bool:
    """ Wait for a servable to become SERVING """
    events_stream = cluster.request("GET", "/api/v2/events", stream=True)
    events_client = sseclient.SSEClient(events_stream)
    status = Servable.find_by_name(cluster, servable_name).status
    if not status is ServableStatus.STARTING and \
            ServableStatus.SERVING:
        return None
    try:
        for event in events_client.events():
            timeout_messages -= 1
            if timeout_messages < 0:
                raise ValueError
            if event.event == "ServableUpdate":
                data = json.loads(event.data)
                if data.get("fullName") == servable_name:
                    status = ServableStatus.from_camel_case(data.get("status", {}).get("status"))
                    if status is ServableStatus.SERVING:
                        return None
                    raise ValueError
    finally:
        events_client.close()
