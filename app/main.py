import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import HeartFeatures, PredictionResponse

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model/heart_model.joblib"))

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts presence of heart disease from 13 clinical features.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API. See /docs for Swagger UI."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "model_type": type(model).__name__,
        "n_estimators": getattr(model, "n_estimators", None),
        "features": FEATURES,
        "target": "heart_disease (true = presence, false = absence)",
        "test_accuracy": round(bundle.get("test_accuracy", 0.0), 3),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HeartFeatures):
    row = pd.DataFrame([features.model_dump()])[FEATURES]
    pred = int(model.predict(row)[0])
    proba = float(model.predict_proba(row)[0][1])
    return PredictionResponse(heart_disease=bool(pred), probability=round(proba, 4))
