# Root Cause Service

Hydro serving models used in demos are stored in [hs_demos](hs_demos).

You can test anchor and rise with `python tests/test_anchor_on_multiple_adult_models.py` and `test/test_rise_on_mobilenet.py` correspondingly.  

## Dependencies

```python
DEBUG_ENV = bool(os.getenv("DEBUG_ENV", True))

REQSTORE_URL = os.getenv("REQSTORE_URL", "managerui:9090")
SERVING_URL = os.getenv("SERVING_URL", "managerui:9090")

MONGO_URL = os.getenv("MONGO_URL", "mongodb")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")
```

`SERVING_URL` is pointing to GRPC service with `ManagerServiceStub`,
`PredictionServiceStub`, `GatewayServiceStub`.