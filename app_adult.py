import json
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import rise
from flask import Flask, request, jsonify
from functools import reduce, partial

from anchor2.anchor2 import TabularExplainer

from reqstore_client import APIHelper, BinaryHelper

from flask import Blueprint

adult_api = Blueprint('adult_api', __name__)


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


def get_adult_data_from_reqstore() -> pd.DataFrame:
    reqstore_url = "https://dev.k8s.hydrosphere.io/reqstore/"
    application_id = "54"  # FIXME remove these hard-coded app_id

    binary_data = APIHelper.download_all(reqstore_url, application_id)
    records = BinaryHelper.decode_records(binary_data)

    all_entries = reduce(list.__add__, map(lambda r: r.entries, records))
    reqstore_requests = map(lambda r: r.request, all_entries)

    # Helper functions to translate requests into pd.Series
    def request_to_df(r):
        columns = []
        values = []
        for key, value in r.inputs.items():
            columns.append(key)
            if len(value.int64_val) == 0:
                values.append([np.NAN])  # FIXME why this field is missing?
            else:
                values.append(value.int64_val)
        return pd.DataFrame(columns=columns, data=np.array(values).T)

    x = list(filter(lambda x: x.shape[0] == 500, map(request_to_df, reqstore_requests)))  # FIXME

    x = pd.concat(x, sort=False)

    # Remove [1, 1, 1, ...] from UI test model calls
    x = x.loc[~np.all(x == np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), axis=1)]
    x.dropna(inplace=True)
    x.drop_duplicates(inplace=True)

    return x


def adult_hydro_serving_classifier_call(x: np.array, adult_feature_names: List[str]):
    service_link = "https://dev.k8s.hydrosphere.io/gateway/application/adult-salary-app"
    response = requests.post(url=service_link, json=make_columnar_json(x, feature_names=adult_feature_names))
    predicted_label = np.array(response.json()["Prediction"])
    return predicted_label


@adult_api.route("/anchor2_adult", methods=['POST'])
def anchor2_adult():
    with open("configs/anchor2_adult.json", "r") as f:
        anchor_config = json.load(f)
    label_decoders: Dict[int, List[str]] = dict([(int(k), v) for (k, v) in anchor_config['label_decoders'].items()])
    # TODO add oh_encoded_categories processing here too
    del anchor_config['label_decoders']

    anchor_explainer = TabularExplainer()
    data = get_adult_data_from_reqstore()
    data = data.loc[:, anchor_config['feature_names']]
    print("Fetched data from reqstore")

    anchor_explainer.fit(data=data,
                         label_decoders=label_decoders,
                         ordinal_features_idx=anchor_config['ordinal_features_idx'],
                         oh_encoded_categories=anchor_config['oh_encoded_categories'],
                         feature_names=anchor_config['feature_names'])

    x = np.array(request.get_json()['sample'][0])

    classifier_fn = partial(adult_hydro_serving_classifier_call, adult_feature_names=anchor_config['feature_names'])
    explanation = anchor_explainer.explain(x, classifier_fn=classifier_fn, )

    return jsonify(explanation=str(explanation), coverage=explanation.coverage(), precision=explanation.precision())
