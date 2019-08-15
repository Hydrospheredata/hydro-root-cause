import os
import sys
from time import sleep

import numpy as np
import pandas as pd
import requests
from hydro_serving_grpc.reqstore import reqstore_client
from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

sys.path.append(f"{SCRIPT_DIR}/..")

from client import HydroServingClient

df = pd.read_csv(f"{SCRIPT_DIR}/../hs_demos/data/test_adult.csv")
print("Loaded adult dataset ", df.shape)

hs_client = HydroServingClient("localhost:9090")
print("RootCause Hello: ", requests.get("http://localhost/rootcause/").text)

# Iterate over 3 possible contract types to check!
for model_name in ['adult_columnar', 'adult_scalar', 'adult_tensor']:
    print(f"{'-' * 40} TESTING MODEL {model_name} {'-' * 40}")
    try:
        adult_model = hs_client.get_model(model_name, 1)
    except Exception as e:
        print("Uploading model....")
        os.system(f"hs cluster use local >/dev/null 2>&1")
        os.system(f"hs upload --dir {SCRIPT_DIR}/../hs_demos/{model_name} >/dev/null 2>&1")
        sleep(10)
        adult_model = hs_client.get_model(model_name, 1)

    print("Model: ", adult_model)

    print("Deploy adult servable and send single sample batches")
    adult_servable = hs_client.deploy_servable(model_name, 1)
    sleep(5)  # Bug - servable is not detected fast enough
    for _ in tqdm(range(10)):
        i = np.random.randint(0, df.shape[0])

        if model_name.endswith("tensor"):
            x = np.array(df.iloc[i])[np.newaxis]
        elif model_name.endswith("scalar"):
            x = df.iloc[i]
        else:
            x = df.iloc[i:(i + 1)]

        output = adult_servable(x, _profile=True)
    print("Example output ", output)
    adult_servable.delete()

    rs_client = reqstore_client.ReqstoreClient("localhost:9090", insecure=True)
    subsample = list(rs_client.subsampling(str(adult_model.id), 10))

    x = adult_model.contract.decode_request(subsample[0].entries[0].request)
    print("Explained sample - ", x)
    folder = adult_model.id
    ts = subsample[0].ts
    uid = subsample[0].entries[0].uid
    print("Explained sample reqstore attributes - ", folder, ts, uid)

    print("Deploy adult servable and send a lot of batches")
    adult_servable = hs_client.deploy_servable(model_name, 1)
    sleep(5)  # Bug - servable is not detected fast enough

    if model_name.endswith("tensor"):
        for i in tqdm(range(df.shape[0] // 100 - 1)):
            x = np.array(df.iloc[i * 100:(i + 1) * 100])
            adult_servable(x, _profile=True)
    elif model_name.endswith("scalar"):
        for i in tqdm(range(df.shape[0])):
            x = df.iloc[i]
            adult_servable(x, _profile=True)
    else:
        for i in tqdm(range(df.shape[0] // 100 - 1)):
            x = df.iloc[i * 100:(i + 1) * 100]
            adult_servable(x, _profile=True)
    adult_servable.delete()

    print("Send rootcause request")
    r = requests.post("http://localhost/rootcause/anchor", json={"model": {"name": model_name, "version": 1},
                                                                 "explained_instance": {"timestamp": ts, "uid": uid}})

    redirect_url = r.headers['Location']
    print("Redirect URL ", redirect_url)

    last_state = ""
    while last_state not in ("SUCCESS", "FAILURE"):
        sleep(5)
        status_response = requests.get(redirect_url).json()
        last_state = status_response['state']

    if last_state == "FAILURE":
        raise Exception(f"Model {model_name} failed!")

    if last_state == "SUCCESS":
        print(f"Model {model_name} got explanation!")
