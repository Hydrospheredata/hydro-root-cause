import json

import numpy as np
import requests
import rise
from flask import Blueprint
from flask import request, jsonify

mnist_api = Blueprint('mnist_api', __name__)


# @mnist_api.route("/mobilenet", methods=['POST'])
# def rise_mobilenet():
#     with open("configs/rise_mobilenetv2.json", "r") as f:
#         rise_config = json.load(f)
#     rise_explainer = rise.RiseImageExplainer()
#
#     prediction_fn = mnist_hydro_serving_classifier_call
#     rise_explainer.fit(prediction_fn=prediction_fn,
#                        input_size=rise_config['input_size'],
#                        class_names=dict([(int(k), v) for (k, v) in rise_config['class_names'].items()]),
#                        number_of_masks=rise_config['number_of_masks'],
#                        mask_granularity=rise_config['mask_granularity'],
#                        mask_density=rise_config['mask_density'],
#                        single_channel=False)
#
#     image = request.get_json()['imgs'][0]
#     image = np.array(image)
#
#     saliency_map = rise_explainer.explain(image)
#     return jsonify(saliency_map.tolist())
#

#
def mnist_hydro_serving_classifier_call(x: np.array):
    service_link = "https://dev.k8s.hydrosphere.io/gateway/application/mnist-app"
    response = requests.post(url=service_link, json={"imgs": x.tolist()})
    predicted_probas = np.array(response.json()["probabilities"])
    return predicted_probas


@mnist_api.route("/mnist", methods=['POST'])
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
