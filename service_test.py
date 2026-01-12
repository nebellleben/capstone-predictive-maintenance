"""
Simplified FastAPI web service for testing - matches the quick model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import List
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Aircraft Predictive Maintenance API (Test)",
    description="Simplified API for testing RUL predictions",
    version="1.0.0-test"
)

# Global variables
model = None
feature_cols = None
sensors_to_exclude = None


# Simplified feature engineering (matches quick training)
def create_simple_features(df, sensors_to_exclude=None):
    """Create simplified features matching the quick training."""
    df = df.copy()
    
    if sensors_to_exclude is None:
        sensors_to_exclude = []
    
    sensor_cols = [f'sensor_{i:02d}' for i in range(1, 22)]
    active_sensors = [col for col in sensor_cols if col not in sensors_to_exclude]
    
    # Add rolling features for first 5 sensors (same as training)
    df = df.sort_values(['engine_id', 'cycle']).reset_index(drop=True)
    for sensor in active_sensors[:5]:
        df[f'{sensor}_roll3'] = df.groupby('engine_id')[sensor].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
    
    return df


# Pydantic models
class SensorData(BaseModel):
    """Single sensor reading."""
    engine_id: int
    cycle: int
    sensor_01: float = 0.0
    sensor_02: float = 0.0
    sensor_03: float = 0.0
    sensor_04: float = 0.0
    sensor_05: float = 0.0
    sensor_06: float = 0.0
    sensor_07: float = 0.0
    sensor_08: float = 0.0
    sensor_09: float = 0.0
    sensor_10: float = 0.0
    sensor_11: float = 0.0
    sensor_12: float = 0.0
    sensor_13: float = 0.0
    sensor_14: float = 0.0
    sensor_15: float = 0.0
    sensor_16: float = 0.0
    sensor_17: float = 0.0
    sensor_18: float = 0.0
    sensor_19: float = 0.0
    sensor_20: float = 0.0
    sensor_21: float = 0.0


class PredictionResponse(BaseModel):
    """Prediction response."""
    engine_id: int
    cycle: int
    predicted_rul: float
    message: str = "Prediction successful"


@app.on_event("startup")
async def load_model():
    """Load model and metadata on startup."""
    global model, feature_cols, sensors_to_exclude
    
    print("Loading model and preprocessing information...")
    model = joblib.load('model.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    sensors_to_exclude = joblib.load('sensors_to_exclude.pkl')
    
    print(f"✓ Model loaded: {type(model).__name__}")
    print(f"✓ Number of features: {len(feature_cols)}")
    print(f"✓ Features: {feature_cols}")
    if sensors_to_exclude:
        print(f"✓ Excluded sensors: {sensors_to_exclude}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Aircraft Predictive Maintenance API (Test Version)",
        "version": "1.0.0-test",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": type(model).__name__,
        "num_features": len(feature_cols)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(sensor_data: SensorData):
    """Predict RUL for a single engine cycle."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        data_dict = sensor_data.dict()
        df = pd.DataFrame([data_dict])
        
        # Apply simplified feature engineering
        df_features = create_simple_features(df, sensors_to_exclude=sensors_to_exclude)
        
        # Select features and handle missing
        X = df_features[feature_cols].fillna(0) if all(col in df_features.columns for col in feature_cols) else df_features[feature_cols].reindex(columns=feature_cols, fill_value=0)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Ensure non-negative
        prediction = max(0, prediction)
        
        return PredictionResponse(
            engine_id=sensor_data.engine_id,
            cycle=sensor_data.cycle,
            predicted_rul=round(prediction, 2),
            message="Prediction successful"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    print("Starting Aircraft Predictive Maintenance API (Test Version)...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
