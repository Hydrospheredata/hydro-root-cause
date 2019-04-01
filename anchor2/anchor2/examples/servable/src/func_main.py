import hydro_serving_grpc as hs
from joblib import load
import numpy as np


clf = load('/model/files/random-forest-adult.joblib')
features = ['Age', 'Workclass', 'Education', 'Marital Status', 'Occupation',
            'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss',
            'Hours per week', 'Country']
            

def extract_value(proto):
    return np.array(proto.int64_val, dtype='int64') \
        .reshape([dim.size for dim in proto.tensor_shape.dim])


def predict(**kwargs):
    extracted = [extract_value(kwargs[feature]) for feature in features]
    transformed = np.squeeze(np.dstack(extracted))
    predicted = clf.predict(transformed)

    response = hs.TensorProto(
        int64_val=predicted.flatten().tolist(),
        dtype=hs.DT_INT64,
        tensor_shape=hs.TensorShapeProto(
            dim=[hs.TensorShapeProto.Dim(size=item) for item in predicted.shape]))
    
    return hs.PredictResponse(outputs={"Prediction": response})