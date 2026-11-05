import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import HeartFeatures, InfoResponse, PredictionResponse

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model/heart_model.joblib"))

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES: list[str] = bundle["features"]

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts presence of heart disease from 13 clinical features.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API. See /docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info", response_model=InfoResponse)
def info():
    return InfoResponse(
        model_type=type(model).__name__,
        features=FEATURES,
        test_accuracy=bundle.get("test_accuracy"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(data: HeartFeatures):
    row = pd.DataFrame([data.model_dump()])[FEATURES]
    proba = float(model.predict_proba(row)[0][1])
    return PredictionResponse(heart_disease=proba >= 0.5, probability=round(proba, 4))
