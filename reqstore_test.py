import time
import numpy as np
import pandas as pd

from hydro_serving_grpc.timemachine.reqstore_client import *


def request_to_df(r):
    columns = []
    values = []
    for key, value in r.inputs.items():
        columns.append(key)
        values.append(value.int64_val)
    return pd.DataFrame(columns=columns, data=np.array(values).T)


client = ReqstoreClient("dev.k8s.hydrosphere.io:443", False)
folder = "54"
end_time = round(time.time() * 1000000000)
data = client.getRange(0, end_time, folder)
data = list(data)

requests = list(map(lambda x: x.entries[0].request, data))
requests = [request_to_df(r) for r in requests]

df = pd.concat(requests, sort=False)

# Remove [1, 1, 1, ...] which results from UI test model calls
df = df.loc[~np.all(df == np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), axis=1)]
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print(df.shape)
