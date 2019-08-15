import sys
from time import sleep

sys.path.append("../")

import numpy as np
import requests
from hydro_serving_grpc.reqstore import reqstore_client
from PIL import Image
from imgcat import imgcat
from client import HydroServingClient

#
hs_client = HydroServingClient("localhost:9090")
mobile_net_035 = hs_client.get_model("mobilenet_v2_035", 1)
print("mobilenet_v2_035: ", mobile_net_035)

print("RootCause Hello: ", requests.get("http://localhost/rootcause/").text)

img = Image.open("../hs_demos/mobilenet_v2_035/demo_image.jpg")
img = img.resize((224, 224))

print("Explained image:")
imgcat(img)

img_arr = np.array(img, dtype=np.float)[np.newaxis]

print("Deploy mobilenet servable and send our image")
mobilenet_servable = hs_client.deploy_servable("mobilenet_v2_035", 1)
try:
    sleep(1)
    mobilenet_servable(img_arr, _profile=True)
except:
    mobilenet_servable.delete()

mobilenet_servable.delete()

rs_client = reqstore_client.ReqstoreClient("localhost:9090", insecure=True)
subsample = list(rs_client.subsampling(str(mobile_net_035.id), 10))

x = mobile_net_035.contract.decode_request(subsample[0].entries[0].request)

folder = mobile_net_035.id
ts = subsample[0].ts
uid = subsample[0].entries[0].uid
print("Explained sample reqstore attributes - ", folder, ts, uid)

print("Send rootcause request")
r = requests.post("http://localhost/rootcause/rise", json={"model": {"name": "mobilenet_v2_035", "version": 1},
                                                           "explained_instance": {"timestamp": ts, "uid": uid}})

redirect_url = r.headers['Location']
print("Redirect URL ", redirect_url)
while True:
    sleep(5)
    print(requests.get(redirect_url).text)
