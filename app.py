from functools import partial

import requests
from typing import Dict, Any, List

from flask import Flask, request, abort, jsonify
import numpy as np
import anchor2
import rise
import json
import os

app = Flask(__name__)

host_address = os.environ.get("CLUSTER_ADDRESS", "https://dev.k8s.hydrosphere.io")
application_name = os.environ.get("APPLICATION_NAME", "adult-salary-app")
signature_name = os.environ.get("SIGNATURE_NAME", "predict")
service_link = f"{host_address}/gateway/applications/{application_name}/{signature_name}"


def hydro_serving_classifier_call(x: np.array, json_tag="samples"):
    response = requests.post(url=service_link, json={json_tag: [x.tolist()]})
    predicted_probas = np.array(response.json()["class_ids"][0])
    return predicted_probas.argmax()


@app.route("/", methods=['POST'])
def main():
    type = request.args.get('type')
    if type is None:
        abort(400, "Specify type of input data")
    elif type == "img":
        try:

            with open("configs/rise_mnist.json", "r") as f:
                rise_config = json.load(f)
            rise_explainer = rise.RiseImageExplainer()

            prediction_fn = hydro_serving_classifier_call  # TODO Complete this with a call to hydroserving
            rise_explainer.fit(prediction_fn=prediction_fn,
                               input_size=rise_config['input_size'],
                               class_names=dict([(int(k), v) for (k, v) in rise_config['class_names'].items()]),
                               number_of_masks=rise_config['number_of_masks'],
                               mask_granularity=rise_config['mask_granularity'],
                               mask_density=rise_config['mask_density'])

            image = request.get_json()['imgs'][0]
            image = np.array(image)

            saliency_map = rise_explainer.explain(image)
            return jsonify(saliency_map.tolist())
        except Exception as exp:
            abort(500, str(exp))
    elif type == "tabular":
        try:
            with open("configs/anchor2_adult.json", "r") as f:
                anchor_config = json.load(f)
            label_decoders: Dict[int, List[str]] = dict([(int(k), v) for (k, v) in anchor_config['label_decoders'].items()])
            del anchor_config['label_decoders']

            anchor_explainer = anchor2.anchor2.TabularExplainer()
            data = None  # TODO Get data from reqstore
            anchor_explainer.fit(data=data, label_decoders=label_decoders, **anchor_config)

            x = request.get_json()['samples'][0]

            classifier_fn = partial(hydro_serving_classifier_call, json_tag='samples')  # TODO Complete this with a call to hydroserving

            explanation = anchor_explainer.explain(x, classifier_fn=classifier_fn, )
            return jsonify(explanation=str(explanation))
        except Exception as exp:
            abort(500, str(exp))
    else:
        abort(400, "Invalid type of input data")



