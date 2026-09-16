import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI application
app = FastAPI(
    title="California Housing Price Prediction API",
    description="Inference service for California Housing XGBoost Regressor",
    version="1.0.0",
)

# Resolve artifact paths relative to the project structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "preprocessor.joblib")
MODEL_PATH = os.path.join(PROJECT_ROOT, "Models", "xgboost_model.joblib")

# Load artifacts into memory on server boot
preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)


# Define request schema matching input features
class HousingInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "California Housing Price Prediction API",
    }


@app.post("/predict")
def predict_housing_price(payload: HousingInput):
    # Convert input payload into DataFrame
    input_df = pd.DataFrame([payload.model_dump()])

    # Compute engineered features identical to SRC/train.py
    input_df["rooms_per_household"] = (
        input_df["total_rooms"] / input_df["households"]
    )
    input_df["bedrooms_per_room"] = (
        input_df["total_bedrooms"] / input_df["total_rooms"]
    )
    input_df["population_per_household"] = (
        input_df["population"] / input_df["households"]
    )
    input_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Transform features with preprocessor artifact
    processed_features = preprocessor.transform(input_df)

    # Predict log-transformed value and map back to dollar scale
    log_prediction = model.predict(processed_features)
    dollar_prediction = float(np.expm1(log_prediction)[0])

    return {"predicted_house_value": round(dollar_prediction, 2)}