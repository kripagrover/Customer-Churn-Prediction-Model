"""Load trained artifacts and run churn predictions."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "customer_churn_model.pkl"
ENCODERS_PATH = PROJECT_ROOT / "encoders.pkl"


def model_files_exist() -> bool:
    return MODEL_PATH.exists() and ENCODERS_PATH.exists()


def load_artifacts() -> tuple[Any, list[str], dict[str, Any]]:
    with open(MODEL_PATH, "rb") as file:
        model_data = pickle.load(file)

    with open(ENCODERS_PATH, "rb") as file:
        encoders = pickle.load(file)

    model = model_data["model"]
    feature_names: list[str] = model_data["features_names"]
    return model, feature_names, encoders


def predict_churn(customer_data: dict[str, Any]) -> dict[str, Any]:
    model, feature_names, encoders = load_artifacts()

    input_df = pd.DataFrame([customer_data])

    for column, encoder in encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])

    input_df = input_df[feature_names]
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    prob_no_churn = float(probabilities[0])
    prob_churn = float(probabilities[1])

    return {
        "prediction": int(prediction),
        "label": "Churn" if prediction == 1 else "No Churn",
        "prob_no_churn": prob_no_churn,
        "prob_churn": prob_churn,
        "risk_level": _risk_level(prob_churn),
    }


def _risk_level(churn_probability: float) -> str:
    if churn_probability >= 0.7:
        return "High"
    if churn_probability >= 0.4:
        return "Medium"
    return "Low"
