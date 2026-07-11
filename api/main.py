import os
import sys
import time
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_engineering import engineer_features  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-disease-api")

PREDICT_REQUESTS = Counter(
    "predict_requests_total", "Total /predict requests", ["outcome"]
)
PREDICT_LATENCY = Histogram(
    "predict_latency_seconds", "Time spent handling /predict requests"
)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib"),
)

app = FastAPI(title="Heart Disease Risk API", version="1.0.0")

model = None


class PatientFeatures(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model not found at {MODEL_PATH}. /predict will fail until trained.")
        return
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(features: PatientFeatures):
    start = time.perf_counter()
    if model is None:
        PREDICT_REQUESTS.labels(outcome="model_unavailable").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info(f"Prediction request: {features.model_dump()}")
    df = pd.DataFrame([features.model_dump()])
    df = engineer_features(df)
    pred = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[0][1])
    response = {"prediction": pred, "risk_probability": round(proba, 4)}
    logger.info(f"Prediction response: {response}")

    PREDICT_REQUESTS.labels(outcome="success").inc()
    PREDICT_LATENCY.observe(time.perf_counter() - start)
    return response
