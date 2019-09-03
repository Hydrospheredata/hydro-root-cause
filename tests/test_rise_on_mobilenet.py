import os
import sys
import warnings
from time import sleep

from hydro_serving_grpc.reqstore import reqstore_client

warnings.simplefilter(action='ignore', category=FutureWarning)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

sys.path.append(f"{SCRIPT_DIR}/..")

import numpy as np
import requests
from imgcat import imgcat
from client import HydroServingClient
from keras.applications.mobilenet_v2 import decode_predictions
import matplotlib.pyplot as plt
from keras.preprocessing import image

import matplotlib

matplotlib.use("module://imgcat")

MODEL_NAME = "mobilenet_v2"

hs_client = HydroServingClient("localhost:9090")

try:
    mobile_net_035 = hs_client.get_model(MODEL_NAME, 1)
except Exception as e:
    print(e)
    print("Uploading model....")
    os.system(f"hs cluster use local >/dev/null 2>&1")
    os.system(f"hs upload --dir {SCRIPT_DIR}/../hs_demos/mobilenet_v2 >/dev/null 2>&1")
    sleep(10)
    mobile_net_035 = hs_client.get_model(MODEL_NAME, 1)

print("Model: ", mobile_net_035)

print("RootCause Hello: ", requests.get("http://localhost/rootcause/").text)


def load_img(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return img, x


img, img_arr = load_img(f"{SCRIPT_DIR}/../rise/examples/bullcar_crop.png")
img_arr = img_arr.astype(np.float64)

print("Explained image:")
imgcat(img)

print("Deploy mobilenet servable and send our image")
mobilenet_servable = hs_client.deploy_servable(MODEL_NAME)
try:
    sleep(5)
    prediction = mobilenet_servable(img_arr, _profile=True)['probabilities']
finally:
    mobilenet_servable.delete()

rs_client = reqstore_client.ReqstoreClient("localhost:9090", insecure=True)
subsample = list(rs_client.subsampling(str(mobile_net_035.id), 10))

x = mobile_net_035.contract.decode_request(subsample[0].entries[0].request)

folder = mobile_net_035.id
ts = subsample[0].ts
uid = subsample[0].entries[0].uid
print("Explained sample reqstore attributes - ", folder, ts, uid)

print("Send rootcause request")
r = requests.post("http://localhost/rootcause/rise", json={"model": {"name": MODEL_NAME, "version": 1},
                                                           "explained_instance": {"timestamp": ts, "uid": uid}})

redirect_url = r.headers['Location']
print("Redirect URL ", redirect_url)
last_state = ""
while last_state not in ("SUCCESS", "FAILURE"):
    sleep(5)
    status_response = requests.get(redirect_url).json()
    print(status_response)
    last_state = status_response['state']

    if last_state == "FAILURE":
        raise Exception(f"Model failed! - {status_response['description']}")

    if last_state == "SUCCESS":
        print(f"Model got explanation!")
        result_id = status_response['result']

        result = requests.get(f"http://localhost/rootcause/fetch_result/rise/{result_id}")

        fig, axes = plt.subplots(1, 5)
        fig.set_size_inches(15, 6)

        decoded_prediction = decode_predictions(prediction)
        print(decoded_prediction)
        print(len(decoded_prediction))

        for i in range(5):
            x = result.json()['result'][i]
            mask = x['mask']
            mask_img = np.array(mask).reshape((224, 224)) / 255

            proba = decoded_prediction[0][i][2]
            decoded_class = decoded_prediction[0][i][1]

            axes[i].set_title(f'Top {i+1}: "{decoded_class}" {proba * 100: .1f}%')
            axes[i].axis('off')
            axes[i].imshow(img)
            axes[i].imshow(mask_img, cmap='jet', alpha=0.4)
        plt.tight_layout()
        imgcat(fig)
