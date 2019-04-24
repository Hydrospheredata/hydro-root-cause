import time
from functools import partial
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from hydro_serving_grpc.timemachine import ReqstoreClient

from anchor2.anchor2 import TabularExplainer
from rise.rise import RiseImageExplainer

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello!"


# Function to store sample in a json with columnar signature
def make_columnar_json(sample, feature_names):
    output_json = {}
    if type(sample) == pd.Series:
        feature_names = sample.index
        sample = sample.values
    elif feature_names is None:
        raise ValueError

    for feature_idx, fname in enumerate(feature_names):
        output_json[fname] = [int(v) for v in sample[:, feature_idx]]
    return output_json


def get_data_from_reqstore(application_id, reqstore_url="dev.k8s.hydrosphere.io:443"):
    def request_to_df(r):
        columns = []
        values = []
        for key, value in r.inputs.items():
            columns.append(key)
            values.append(value.int64_val)
        return pd.DataFrame(columns=columns, data=np.array(values).T)

    client = ReqstoreClient(reqstore_url, False)
    end_time = round(time.time() * 1000000000)
    data = client.getRange(0, end_time, application_id)
    data = list(data)

    rs = list(map(lambda x: x.entries[0].request, data))
    rs = [request_to_df(r) for r in rs]

    df = pd.concat(rs, sort=False)

    # Remove [1, 1, 1, ...] which results from UI test model calls
    df = df.loc[~np.all(df == np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), axis=1)]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    return df


def get_application_id(application_name: str):
    service_link = "https://dev.k8s.hydrosphere.io/api/v2/application/"
    r = requests.get(service_link + application_name)
    return r.json()['executionGraph']['stages'][0]['modelVariants'][0]['modelVersion']["id"]


def hydroserving_classifier_call(x, feature_names, application_name,
                                 service_link="https://dev.k8s.hydrosphere.io/gateway/application/",
                                 return_label="Prediction"):
    response = requests.post(url=service_link + application_name, json=make_columnar_json(x, feature_names=feature_names))
    predicted_label = np.array(response.json()[return_label])
    return predicted_label


def is_valid_anchor_config(anchor_config):
    return True


@app.route("/anchor", methods=['POST'])
def anchor():
    inp_json = request.get_json()
    # TODO verify input json
    application_id = get_application_id(inp_json['application_name'])
    anchor_config = inp_json['config']
    data = get_data_from_reqstore(str(application_id))
    print("DATA SHAPE ", data.shape)

    label_decoders: Dict[int, List[str]] = dict([(int(k), v) for (k, v) in anchor_config['label_decoders'].items()])
    oh_encoded_categories: Dict[str, List[int]] = dict(
        [(k, [int(v) for v in vs]) for (k, vs) in anchor_config['oh_encoded_categories'].items()])

    anchor_explainer = TabularExplainer()
    data = data.loc[:, anchor_config['feature_names']]  # Sort columns according to feature names order
    anchor_explainer.fit(data=data,
                         label_decoders=label_decoders,
                         ordinal_features_idx=anchor_config['ordinal_features_idx'],
                         oh_encoded_categories=oh_encoded_categories,
                         feature_names=anchor_config['feature_names'])

    x = np.array(request.get_json()['explained_instance'])

    classifier_fn = partial(hydroserving_classifier_call,
                            feature_names=anchor_config['feature_names'],
                            application_name=inp_json['application_name'])

    explanation = anchor_explainer.explain(x, classifier_fn=classifier_fn)

    return jsonify(explanation=str(explanation), coverage=explanation.coverage(), precision=explanation.precision())


def hydroserving_image_classifier_call(x,
                                       application_name,
                                       service_link="https://dev.k8s.hydrosphere.io/gateway/application/"):
    url = service_link + application_name
    response = requests.post(url=url, json={"imgs": x.tolist()})
    predicted_probas = np.array(response.json()["probabilities"])
    return predicted_probas


def is_valid_rise_config(rise_config):
    return True


@app.route("/rise", methods=['POST'])
def rise():
    inp_json = request.get_json()
    rise_config = inp_json['config']
    # TODO check config

    rise_explainer = RiseImageExplainer()

    prediction_fn = partial(hydroserving_image_classifier_call,
                            application_name=rise_config['application_name'])

    rise_explainer.fit(prediction_fn=prediction_fn,
                       input_size=rise_config['input_size'],
                       number_of_masks=rise_config['number_of_masks'],
                       mask_granularity=rise_config['mask_granularity'],
                       mask_density=rise_config['mask_density'],
                       single_channel=rise_config['single_channel'])

    image = request.get_json()['image']
    image = np.array(image)

    saliency_map = rise_explainer.explain(image)
    return jsonify(saliency_map.tolist())


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
