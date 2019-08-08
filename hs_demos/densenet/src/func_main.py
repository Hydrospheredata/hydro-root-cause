import hydro_serving_grpc as hs
import numpy as np
import tensorflow as tf
from keras.applications import DenseNet201


def extract_value(proto):
    return np.array(proto.double_val, dtype='float64').reshape((-1, 224, 224, 3))


dnet = DenseNet201(weights="/model/files/densenet201_weights_tf_dim_ordering_tf_kernels.h5")
# dnet = DenseNet201()

global graph
graph = tf.get_default_graph()


def predict(**kwargs):
    images = extract_value(kwargs['input'])

    with graph.as_default():
        probas = dnet.predict(images)

    classes = probas.argmax(axis=1)

    probas_proto = hs.TensorProto(
        double_val=probas.flatten().tolist(),
        dtype=hs.DT_DOUBLE,
        tensor_shape=hs.TensorShapeProto(
            dim=[hs.TensorShapeProto.Dim(size=-1), hs.TensorShapeProto.Dim(size=1000)]))

    classes_proto = hs.TensorProto(
        int64_val=classes.flatten().tolist(),
        dtype=hs.DT_INT64,
        tensor_shape=hs.TensorShapeProto(
            dim=[hs.TensorShapeProto.Dim(size=-1), hs.TensorShapeProto.Dim(size=1)]))

    return hs.PredictResponse(outputs={"classes": classes_proto, "probabilities": probas_proto})
