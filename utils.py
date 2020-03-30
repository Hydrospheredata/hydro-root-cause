from enum import Enum
from typing import Tuple

from hydro_serving_grpc.contract import ModelSignature


class AlwaysTrueObj(object):
    def __eq__(self, other):
        return True


AnyDimSize = AlwaysTrueObj()


def contract_is_supported_by_rise(signature: ModelSignature, training_data_available) -> Tuple[bool, str]:
    # TODO add pydoc here and to README
    # TODO check for tensor profiling type (?)

    # Rise supports batch and non\batch request of images of grayscale or 3 channeled.
    rise_supported_input_shapes = [(-1, AnyDimSize, AnyDimSize, 3),
                                   (-1, AnyDimSize, AnyDimSize, 1),
                                   (-1, AnyDimSize, AnyDimSize), ]

    # Right now single point API is not supported, tech debt in rise_task
    # (AnyDimSize, AnyDimSize, 3),
    # (AnyDimSize, AnyDimSize, 1),
    # (AnyDimSize, AnyDimSize)]

    rise_supported_probabilities_shapes = [(-1, AnyDimSize), ]
    # (AnyDimSize,)]

    if 'probabilities' in [tensor.name for tensor in signature.outputs]:
        probabilities_field = next(filter(lambda t: t.name == "probabilities", signature.outputs))
        probabilities_shape = tuple(map(lambda dim: dim.size, probabilities_field.shape.dim))
        if probabilities_shape not in rise_supported_probabilities_shapes:
            return False, f"Shape {probabilities_shape} of probabilities tensor is not supported"
    else:
        return False, "No field named 'probabilities' in contract signature was found"

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


class TabularContractType(Enum):
    INVALID = -1
    SCALAR = 0
    COLUMNAR = 1
    SINGLE_TENSOR = 2


def get_tabular_contract_type(signature: ModelSignature) -> TabularContractType:
    input_tensor_shapes = [tuple(map(lambda dim: dim.size, input_tensor.shape.dim)) for input_tensor in signature.inputs]

    if len(input_tensor_shapes) == 1 and 1 <= len(input_tensor_shapes[0]) <= 2:
        return TabularContractType.SINGLE_TENSOR
    elif all([shape == tuple() for shape in input_tensor_shapes]):
        return TabularContractType.SCALAR
    elif all([shape == (-1, AnyDimSize) for shape in input_tensor_shapes]):
        return TabularContractType.COLUMNAR
    else:
        return TabularContractType.INVALID


def contract_is_supported_by_anchor(signature, training_data_available) -> Tuple[bool, str]:
    # TODO check for tensor profiling type (?)

    if not training_data_available:
        return False, "No training data available. Provide training data"

    anchor_supported_classes_shapes = [(-1, AnyDimSize),
                                       (AnyDimSize,),
                                       tuple()]

    if 'classes' in [tensor.name for tensor in signature.outputs]:
        classes_field = next(filter(lambda t: t.name == "classes", signature.outputs))
        classes_field_shape = tuple(map(lambda dim: dim.size, classes_field.shape.dim))
        if classes_field_shape not in anchor_supported_classes_shapes:
            return False, f"Classes tensor is of unsupported shape - {classes_field_shape}"
    else:
        return False, "No 'classes' tensor is present"

    tabular_signature_type = get_tabular_contract_type(signature)

    if tabular_signature_type == TabularContractType.INVALID:
        return False, "This signature has invalid type"
    elif tabular_signature_type == TabularContractType.SINGLE_TENSOR:
        return False, "Do not support single tensor contract for now"
    else:
        return True, "Everything is fine"


def get_method_support_statuses(signature: ModelSignature, training_data_available: bool):
    # TODO check for name of explained tensor name!
    method_support_statuses = []

    for method_name, is_supported in (("anchor", contract_is_supported_by_anchor), ("rise", contract_is_supported_by_rise)):
        is_supported, msg = is_supported(signature, training_data_available)
        if is_supported:
            method_support_statuses.append({"name": method_name, "supported": True})
        else:
            method_support_statuses.append({"name": method_name, "supported": False, "msg": msg})

    return method_support_statuses


def get_default_config(method):
    """
    Default configuration for each interpretability method
    :param method:
    :return: dictionary with default parameters
    """
    # TODO check default config to match OpenAPI schema
    if method == "rise":
        return {"number_of_masks": 100,
                "mask_granularity": 100,
                "mask_density": 100}
    elif method == "anchor":

        # Convert config dicts to appropriate types
        # label_decoders: Dict[int, List[str]] = dict([(int(k), v) for (k, v) in config['label_decoders'].items()])
        # label_decoders: Dict[int, List[str]] = dict()  # This is a temp workaround, since we can't pass this config through UI

        # oh_encoded_categories: Dict[str, List[int]] = dict(
        # [(k, [int(v) for v in vs]) for (k, vs) in config['oh_encoded_categories'].items()])
        # oh_encoded_categories: Dict[str, List[int]] = dict()  # This is a temp workaround, since we can't pass this config through UI

        return {"output_explained_tensor_name": "classes",
                "training_subsample_size": 10000,
                "ordinal_features_idx": [0, 11],
                "precision_threshold": 0.5,
                "label_decoders": {},
                "oh_encoded_categories": {},
                "anchor_pool_size": 10,
                "beam_size": 5,
                "batch_size": 100,
                "tolerance": 0.35,
                "delta": 0.25}
    else:
        raise ValueError(f"Invalid method {method} passed")
