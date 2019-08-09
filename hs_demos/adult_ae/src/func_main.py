import hydro_serving_grpc as hs
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import CustomObjectScope


def extract_value(proto):
    return np.array(proto.double_val, dtype='float64').reshape((-1, 12))


maxs = np.array([90., 8., 6., 3., 8., 5., 4., 1., 2., 2., 99., 10.])
mins = np.array([17., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])

with CustomObjectScope({'maxs': maxs, 'mins': mins}):
    m = load_model("/model/files/adult_ae", compile=False)

global graph
graph = tf.get_default_graph()


def predict(**kwargs):
    x = extract_value(kwargs['input'])

    with graph.as_default():
        x_reconstr = m.predict(x)

    losses = np.mean(np.square(x - x_reconstr), axis=1)
    losses_proto = hs.TensorProto(
        double_val=losses.flatten().tolist(),
        dtype=hs.DT_DOUBLE,
        tensor_shape=hs.TensorShapeProto(
            dim=[hs.TensorShapeProto.Dim(size=-1), hs.TensorShapeProto.Dim(size=1)]))

    return hs.PredictResponse(outputs={"ae_loss": losses_proto})
