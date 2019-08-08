import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from time import sleep
from hydro_serving_grpc.reqstore import reqstore_client

import hydro_serving_grpc as hs_grpc
from client import HydroServingClient, HydroServingServable, HydroServingModel

hs_client = HydroServingClient("localhost:9090")
adult_model = hs_client.get_model("adult-tensor", 1)
print("adult_model: ", adult_model)

print("RootCause Hello: ", requests.get("http://localhost/rootcause/").text)

df = pd.read_csv("hs_demos/data/test_adult.csv")
print("Loaded adult dataset ", df.shape)


print("Deploy adult_servable and send single sample batches")
adult_servable = hs_client.deploy_servable("adult-tensor", 1)
sleep(1)  # Bug - servable is not detected fast enough
for _ in tqdm(range(10)):
    sleep(0.2)
    i = np.random.randint(0, df.shape[0])
    x = np.array(df.iloc[i])[np.newaxis]
    adult_servable(x, _profile=True)
adult_servable.delete()

rs_client = reqstore_client.ReqstoreClient("localhost:9090", insecure=True)
subsample = list(rs_client.subsampling(str(adult_model.id), 10))
x = adult_model.contract.decode_request(subsample[0].entries[0].request)
print("Explained sample - ", x)
folder = adult_model.id
ts = subsample[0].ts
uid = subsample[0].entries[0].uid
print("Explained sample reqstore attributes - ", folder, ts, uid)

print("Deploy adult_servable and send a lot of batches")
adult_servable = hs_client.deploy_servable("adult-tensor", 1)
sleep(1)  # Bug - servable is not detected fast enough
for i in tqdm(range(df.shape[0]//100 - 1)):
    sleep(0.2)
    x = np.array(df.iloc[i*100:(i+1)*100])
    adult_servable(x, _profile=True)
adult_servable.delete()

print("Send rootcause request")
r = requests.post("http://localhost/rootcause/anchor", json={"model":{"name":"adult-tensor", "version":1},
                                                      "explained_instance":{"timestamp":ts, "uid": uid}})


redirect_url = r.headers['Location']
print("Redirect URL ", redirect_url)
while True:
	sleep(5)
	print(requests.get(redirect_url).text)