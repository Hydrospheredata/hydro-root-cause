import time
from typing import Union

import hydro_serving_grpc as hs
import numpy as np
import pandas as pd
import requests
import os
import grpc

from google.protobuf.json_format import MessageToDict
from hydro_serving_grpc import manager as hsm
from hydro_serving_grpc.timemachine import ReqstoreClient

# REQSTORE_URL = os.getenv("REQSTORE_URL")
# REQSTORE_PORT = os.getenv("REQSTORE_PORT")
#
# SERVING_URL = os.getenv("SERVING_URL", "http://fc13d681.serving.odsc.k8s.hydrosphere.io")
# SERVING_NOSONAR_PORT = os.getenv("SERVING_NOSONAR_PORT", 9090)
#
# channel = grpc.insecure_channel(f'{SERVING_URL}:{SERVING_NOSONAR_PORT}')
# hsg_client = hs.gateway.GatewayServiceStub(channel)

#
# def check_tabular_contract(model_name: str):
#     model_id = get_application_id(model_name)
#     # __model_spec.version.value = 1
#     r = hsm.GetVersionRequest(id=model_id)
#
#     client = hsm.ManagerServiceStub(channel)
#
#     model_version = MessageToDict(client.GetVersion(r))
#
#     contract = model_version['contract']
#     inputs = contract['predict']['inputs']
#     outputs = contract['predict']['outputs']
#
#     assert "prediction" in [o['model_name'] for o in outputs], "Something should be returned"
#     # TODO verify prediction shape
#     batch_dim_present = False
#
#     shapes = [tuple([int(d['size']) for d in i['shape']['dim']]) for i in inputs]
#     output_shapes = [tuple([int(d['size']) for d in i['shape']['dim']]) for i in outputs]
#
#     dtypes = [i['dtype'] for i in inputs]
#     feature_names = [i['model_name'] for i in inputs]
#     # TODO assert dtypes!
#
#     # Check for presences batch_dim
#     if any([s[0] == -1 for s in shapes]):
#         assert all([s[0] == -1 for s in shapes]), "All columns must contain batch-dim"
#         batch_dim_present = True
#
#     # Check that data is either 2D (including batch dim) or 1D (without batch dim)
#     if any([len(s) > 2 for s in shapes]) or any([len(s) > 1 for s in shapes]) and not batch_dim_present:
#         raise ValueError("Store tensor data in 2D with batch_dim=-1 or either 1D")
#
#     # Check that data is either series of scalar, [[-1, 1], [-1, 1], ...], [-1, X] or just [X]
#     is_column_defined = all([s == (-1, 1) for s in shapes]) or all(s == ('scalar',) for s in shapes)
#     is_tensor_defined = len(shapes) == 1
#
#     if not (is_column_defined or is_tensor_defined):
#         raise ValueError("Can not comprehend contract")
#
#     if is_tensor_defined:
#         shape = shapes[0]
#         name = inputs[0]['model_name']
#         del dtypes, shapes
#         final_shape = shape
#     else:
#         final_shape = (-1, len(shapes)) if batch_dim_present else (len(shapes),)
#
#     return final_shape, dtypes, is_column_defined, feature_names
#
#
# def hs_tabular_call(x, model_name):
#     contract_shape, dtypes, is_columnar, feature_names = check_tabular_contract(model_name)
#
#     if x.shape[-1] != contract_shape[-1]:
#         raise ValueError("Contract violation: invalid # of features")
#
#     j = make_proto_input(x, dtypes, feature_names)
#
#     model_spec = hs.ModelSpec(name=model_name)
#
#     r = hs.PredictRequest(model_spec=model_spec, inputs=inputs)
#     result = hsg_client.PredictModelOnly(r)
#     predicted_label = np.array(result["prediction"])
#
#     return predicted_label
#
#
# def hs_img_call(x, application_name):
#     # TODO change to dirka Bulata
#     url = f"{SERVING_URL}/gateway/application/{application_name}"
#     response = requests.post(url=url, json={"imgs": x.tolist()})
#     try:
#         predicted_probas = np.array(response.json()["probabilities"])
#         return predicted_probas
#     except KeyError:
#         print("Probabilities not found", response.text)
#         raise ValueError
#     except ValueError:
#         print(response.text)
#         raise ValueError
#
#
# # Function to store sample in a json with columnar signature
# def make_proto_input(sample: Union[pd.Series, np.ndarray], has_batch_dim, is_columnar, dtypes=None, feature_names=None):
#     if type(sample) == pd.Series:
#         feature_names = list(sample.index)
#         dtypes = list(sample.dtypes)
#         sample = sample.values
#     elif feature_names is None or dtypes is None:
#         raise ValueError("If np.ndarray is passed, provide feature_names and dtypes")
#
#     if sample.shape[1] != len(feature_names) or sample.shape[1] != len(dtypes):
#         raise ValueError("Feature names / Dtypes are inconsistent with instance shape")
#
#     signatured_json = {}
#
#     for feature_idx, fname in enumerate(feature_names):
#         signatured_json[fname] = sample[:, feature_idx]
#
#     keys = signatured_json.keys()
#
#     batch_size = sample.shape[0]
#     if has_batch_dim and is_columnar:
#         tensor_shapes = [hs.TensorShapeProto(dim=[hs.TensorShapeProto.Dim(size=batch_size), hs.TensorShapeProto.Dim(size=1)])
#                          for k in keys]
#     elif not has_batch_dim and is_columnar:
#         tensor_shapes = [hs.TensorShapeProto(dim=[hs.TensorShapeProto.Dim(size=1)]) for k in keys]
#     elif has_batch_dim and not is_columnar:
#         tensor_shapes = hs.TensorShapeProto(dim=[hs.TensorShapeProto.Dim(size=batch_size), hs.TensorShapeProto.Dim(size=sample.shape[1])])
#     elif not has_batch_dim and not is_columnar:
#         tensor_shapes = hs.TensorShapeProto(dim=[hs.TensorShapeProto.Dim(size=sample.shape[1])])
#
#     assert len(tensor_shapes) == len(dtypes)
#     tensors = [hs.TensorProto(dtype=hs.DT_INT64, int64_val=signatured_json[k], tensor_shape=ts) for k, ts in zip(keys, tensor_shapes)]
#
#     output_proto = dict(zip(keys, tensors))
#     return output_proto
#
#
# def get_data_from_reqstore(application_id):
#     # TODO change to subsampling Dima
#     def request_to_df(r):
#         columns = []
#         values = []
#         for key, value in r.inputs.items():
#             columns.append(key)
#             values.append(value.int64_val)
#         return pd.DataFrame(columns=columns, data=np.array(values).T)
#
#     client = ReqstoreClient(REQSTORE_URL, False)
#     end_time = round(time.time() * 1000000000)
#     data = client.getRange(0, end_time, application_id)
#     data = list(data)
#
#     rs = list(map(lambda x: x.entries[0].request, data))
#     rs = [request_to_df(r) for r in rs]
#
#     df = pd.concat(rs, sort=False)
#
#     # Remove [1, 1, 1, ...] which results from UI test model calls
#     df = df.loc[~np.all(df == np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), axis=1)]
#     df.dropna(inplace=True)
#     df.drop_duplicates(inplace=True)
#
#     return df
