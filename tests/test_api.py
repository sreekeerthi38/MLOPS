import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import pytest
from fastapi.testclient import TestClient
import main as api_main

SAMPLE = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3,
    "slope": 0, "ca": 0, "thal": 1,
}


@pytest.fixture
def client():
    # Use as context manager so FastAPI's startup event (model loading) fires.
    with TestClient(api_main.app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_predict_without_model_returns_503_or_ok(client):
    resp = client.post("/predict", json=SAMPLE)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert "prediction" in body and "risk_probability" in body
