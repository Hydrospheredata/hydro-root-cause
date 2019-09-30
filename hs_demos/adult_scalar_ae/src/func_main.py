import hydro_serving_grpc as hs
import numpy as np
from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf

features = ['age',
            'workclass',
            'education',
            'marital_status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital_gain',
            'capital_loss',
            'hours_per_week',
            'country']


def extract_value(proto):
    return np.array(proto.int64_val, dtype='int64')[0]

maxs = np.array([90., 8., 6., 3., 8., 5., 4., 1., 2., 2., 99., 10.])
mins = np.array([17., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])

with CustomObjectScope({'maxs': maxs, 'mins': mins}):
    m = load_model("/model/files/adult_ae", compile=False)

global graph
graph = tf.get_default_graph()


def predict(**kwargs):
    extracted = np.array([extract_value(kwargs[feature]) for feature in features])
    x = np.dstack(extracted).reshape(1, len(features))

    with graph.as_default():
        x_reconstr = m.predict(x)

    losses = np.mean(np.square(x - x_reconstr), axis=1)
    losses_proto = hs.TensorProto(
        double_val=losses.flatten().tolist(),
        dtype=hs.DT_DOUBLE,
        tensor_shape=hs.TensorShapeProto())

    return hs.PredictResponse(outputs={"value": losses_proto})
