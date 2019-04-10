import json
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import rise
from flask import Flask, request, jsonify
from functools import reduce

from anchor2.anchor2 import TabularExplainer

from reqstore_client import APIHelper, BinaryHelper

app = Flask(__name__)


@app.route("/mobilenet", methods=['POST'])
def rise_mobilenet():
    with open("configs/rise_mobilenetv2.json", "r") as f:
        rise_config = json.load(f)
    rise_explainer = rise.RiseImageExplainer()

    prediction_fn = mnist_hydro_serving_classifier_call
    rise_explainer.fit(prediction_fn=prediction_fn,
                       input_size=rise_config['input_size'],
                       class_names=dict([(int(k), v) for (k, v) in rise_config['class_names'].items()]),
                       number_of_masks=rise_config['number_of_masks'],
                       mask_granularity=rise_config['mask_granularity'],
                       mask_density=rise_config['mask_density'],
                       single_channel=False)

    image = request.get_json()['imgs'][0]
    image = np.array(image)

    saliency_map = rise_explainer.explain(image)
    return jsonify(saliency_map.tolist())


def mnist_hydro_serving_classifier_call(x: np.array):
    service_link = "https://dev.k8s.hydrosphere.io/gateway/application/mnist"
    response = requests.post(url=service_link, json={"imgs": x.tolist()})
    predicted_probas = np.array(response.json()["probabilities"])
    return predicted_probas


@app.route("/mnist", methods=['POST'])
def rise_mnist():
    with open("configs/rise_mnist.json", "r") as f:
        rise_config = json.load(f)
    rise_explainer = rise.RiseImageExplainer()

    prediction_fn = mnist_hydro_serving_classifier_call
    rise_explainer.fit(prediction_fn=prediction_fn,
                       input_size=rise_config['input_size'],
                       class_names=dict([(int(k), v) for (k, v) in rise_config['class_names'].items()]),
                       number_of_masks=rise_config['number_of_masks'],
                       mask_granularity=rise_config['mask_granularity'],
                       mask_density=rise_config['mask_density'],
                       single_channel=True)

    image = request.get_json()['imgs'][0]
    image = np.array(image)

    saliency_map = rise_explainer.explain(image)
    return jsonify(saliency_map.tolist())


@app.route("/")
def hello():
    return "Hello!"


def get_data_from_reqstore() -> pd.DataFrame:
    reqstore_url = "https://dev.k8s.hydrosphere.io/reqstore/"
    application_id = "54"  # FIXME remove these hard-coded app_id

    binary_data = APIHelper.download_all(reqstore_url, application_id)
    records = BinaryHelper.decode_records(binary_data)

    all_entries = reduce(list.__add__, map(lambda r: r.entries, records))
    reqstore_requests = map(lambda r: r.request, all_entries)
    reqstore_responses = map(lambda r: r.response, all_entries)

    # Helper functions to translate requests into pd.Series
    def request_to_series(r):
        columns = []
        values = []
        for key, value in r.inputs.items():
            columns.append(key)
            values.append(value.int64_val[0])
        return pd.Series(index=columns, data=values)

    def response_to_series(r):
        value = r.outputs['Prediction'].int64_val[0]
        return value

    x = list(map(request_to_series, reqstore_requests))
    y = list(map(response_to_series, reqstore_responses))

    x = pd.DataFrame(x)

    # Remove [1, 1, 1, ...] from UI test model calls
    x = x.loc[~np.all(x == np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), axis=1)]
    x.drop_duplicates(inplace=True)

    return x


# Function to store sample in a json with signature specified by Ilnur
def make_signatured_json(sample):
    output_json = {}
    if type(sample) == pd.Series:
        feature_names = sample.index
        values = sample.values
    else:
        feature_names = [
            "Age",
            "Workclass",
            "Education",
            "Marital Status",
            "Occupation",
            "Relationship",
            "Race",
            "Sex",
            "Capital Gain",
            "Capital Loss",
            "Hours per week",
            "Country"]
        values = sample
    for feature_idx, fname in enumerate(feature_names):
        output_json[fname] = [int(v) for v in values[:, feature_idx]]
    return output_json


def adult_hydro_serving_classifier_call(x: np.array):
    service_link = "https://dev.k8s.hydrosphere.io/gateway/application/adult-salary-app"
    response = requests.post(url=service_link, json=make_signatured_json(x))
    predicted_label = np.array(response.json()["Prediction"])
    return predicted_label


@app.route("/anchor2_adult", methods=['POST'])
def anchor2_adult():
    with open("configs/anchor2_adult.json", "r") as f:
        anchor_config = json.load(f)
    label_decoders: Dict[int, List[str]] = dict([(int(k), v) for (k, v) in anchor_config['label_decoders'].items()])
    # TODO add oh_encoded_categories processing here too
    del anchor_config['label_decoders']

    anchor_explainer = TabularExplainer()
    data = get_data_from_reqstore()
    data = data.loc[:, anchor_config['feature_names']]
    print("Fetched data from reqstore")

    anchor_explainer.fit(data=data,
                         label_decoders=label_decoders,
                         ordinal_features_idx=anchor_config['ordinal_features_idx'],
                         oh_encoded_categories=anchor_config['oh_encoded_categories'],
                         feature_names=anchor_config['feature_names'])

    x = np.array(request.get_json()['sample'][0])

    classifier_fn = adult_hydro_serving_classifier_call

    explanation = anchor_explainer.explain(x, classifier_fn=classifier_fn, )
    return jsonify(explanation=str(explanation), coverage=explanation.coverage(), precision=explanation.precision())


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
