import os
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Global artifact references
preprocessor = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global preprocessor, model
    try:
        # Load preprocessor
        if os.path.exists("preprocessor.joblib"):
            preprocessor = joblib.load("preprocessor.joblib")
        
        # Load model artifact fallback
        model_path = os.path.join("Models", "xgboost_model.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        
        print("[Lifespan] Preprocessor and Model loaded successfully.")
    except Exception as e:
        print(f"[Lifespan Error] Failed to load artifacts: {e}")
    yield

app = FastAPI(title="California Housing Price Prediction API", lifespan=lifespan)

class HousingInput(BaseModel):
    longitude: float = Field(..., ge=-180.0, le=180.0, json_schema_extra={"example": -122.23})
    latitude: float = Field(..., ge=-90.0, le=90.0, json_schema_extra={"example": 37.88})
    housing_median_age: float = Field(..., ge=0.0, json_schema_extra={"example": 41.0})
    total_rooms: float = Field(..., gt=0.0, json_schema_extra={"example": 880.0})
    total_bedrooms: float = Field(..., gt=0.0, json_schema_extra={"example": 129.0})
    population: float = Field(..., gt=0.0, json_schema_extra={"example": 322.0})
    households: float = Field(..., gt=0.0, json_schema_extra={"example": 126.0})
    median_income: float = Field(..., ge=0.0, json_schema_extra={"example": 8.3252})
    ocean_proximity: str = Field(..., json_schema_extra={"example": "NEAR BAY"})

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "preprocessor_loaded": preprocessor is not None,
        "model_loaded": model is not None,
    }

@app.post("/predict")
def predict(payload: HousingInput):
    if preprocessor is None or model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or Preprocessor is not loaded.",
        )

    # Convert payload to DataFrame
    input_data = pd.DataFrame([payload.model_dump()])

    # Feature engineering matching training steps
    input_data["rooms_per_household"] = input_data["total_rooms"] / input_data["households"]
    input_data["bedrooms_per_room"] = input_data["total_bedrooms"] / input_data["total_rooms"]
    input_data["population_per_household"] = input_data["population"] / input_data["households"]

    # Preprocessing and inference
    transformed_features = preprocessor.transform(input_data)
    y_log_pred = model.predict(transformed_features)
    y_dollar_pred = float(np.expm1(y_log_pred)[0])

    return {
        "prediction_dollar": round(y_dollar_pred, 2),
        "status": "success",
    }