## Anchor. Tabular data
`POST /anchor`

```
adult_anchor_config = {
  "precision_threshold": 0.95,
  "verbose": False,
  "ordinal_features_idx": [
    0,
    10
  ],
  "oh_encoded_categories": {},
  "label_decoders": {
    "1": [
      " ?",
      ...
    ],
    "2": [
      "Associates",
      ...
    ],
    "3": [
      "Married",
      ...
    ],
    ...
  },
  "strategy": "kl-lucb",
  "feature_names": [
    "Age",
    ...
  ]
}

anchor_link = "http://0.0.0.0:5000/anchor" 

requests.post(url=anchor_link, json={"explained_instance": x.tolist(),
                                    "application_name" : "adult-salary-app",
                                    "config" : adult_anchor_config})
                                                
```

## RISE. Image data
`POST /rise`

```
import requests
rise_link = "http://0.0.0.0:5000/rise" 
rise_config = {"number_of_masks":300,
               "mask_granularity":3,
               "mask_density":0.4,
               "input_size": [28,28],
               "single_channel": True,
               "application_name":"mnist-app"}
requests.post(url=rise_link, json={"image": x.tolist(), "config": rise_config}) 
```