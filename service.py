"""
FastAPI web service for Aircraft Predictive Maintenance model.
Provides REST API endpoints for RUL predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Aircraft Predictive Maintenance API",
    description="API for predicting Remaining Useful Life (RUL) of aircraft engines",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and metadata
model = None
feature_cols = None
sensors_to_exclude = None


# Feature engineering function (same as in train.py)
def create_features(df, sensors_to_exclude=None):
    """
    Create engineered features from raw sensor data.
    """
    df = df.copy()
    
    if sensors_to_exclude is None:
        sensors_to_exclude = []
    sensor_cols = [col for col in df.columns if col.startswith('sensor_') and col not in sensors_to_exclude]
    
    df = df.sort_values(['engine_id', 'cycle']).reset_index(drop=True)
    
    # Time-based features
    df['cycle_norm'] = df.groupby('engine_id')['cycle'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1))
    
    # Rolling statistics
    window = 5
    for sensor in sensor_cols:
        df[f'{sensor}_rolling_mean'] = df.groupby('engine_id')[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'{sensor}_rolling_std'] = df.groupby('engine_id')[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
        )
    
    # Degradation indicators
    for sensor in sensor_cols:
        df[f'{sensor}_diff'] = df.groupby('engine_id')[sensor].diff().fillna(0)
        df[f'{sensor}_diff_norm'] = df[f'{sensor}_diff'] / (df['cycle'] + 1)
    
    # Engine-level aggregations
    engine_stats = df.groupby('engine_id')[sensor_cols].agg(['max', 'min', 'mean', 'std']).reset_index()
    engine_stats.columns = ['engine_id'] + [f'{col}_{stat}' for col in sensor_cols for stat in ['max', 'min', 'mean', 'std']]
    df = df.merge(engine_stats, on='engine_id', how='left')
    
    # Sensor interactions
    if len(sensor_cols) >= 4:
        if 'sensor_01' in sensor_cols and 'sensor_02' in sensor_cols:
            df['sensor_ratio_01_02'] = df['sensor_01'] / (df['sensor_02'] + 1e-10)
        if 'sensor_03' in sensor_cols and 'sensor_04' in sensor_cols:
            df['sensor_ratio_03_04'] = df['sensor_03'] / (df['sensor_04'] + 1e-10)
    
    return df


# Pydantic models for request/response
class SensorData(BaseModel):
    """Single sensor reading."""
    engine_id: int = Field(..., description="Engine unit ID")
    cycle: int = Field(..., description="Cycle number")
    sensor_01: float = Field(..., alias="sensor_01")
    sensor_02: float = Field(..., alias="sensor_02")
    sensor_03: float = Field(..., alias="sensor_03")
    sensor_04: float = Field(..., alias="sensor_04")
    sensor_05: float = Field(..., alias="sensor_05")
    sensor_06: float = Field(..., alias="sensor_06")
    sensor_07: float = Field(..., alias="sensor_07")
    sensor_08: float = Field(..., alias="sensor_08")
    sensor_09: float = Field(..., alias="sensor_09")
    sensor_10: float = Field(..., alias="sensor_10")
    sensor_11: float = Field(..., alias="sensor_11")
    sensor_12: float = Field(..., alias="sensor_12")
    sensor_13: float = Field(..., alias="sensor_13")
    sensor_14: float = Field(..., alias="sensor_14")
    sensor_15: float = Field(..., alias="sensor_15")
    sensor_16: float = Field(..., alias="sensor_16")
    sensor_17: float = Field(..., alias="sensor_17")
    sensor_18: float = Field(..., alias="sensor_18")
    sensor_19: float = Field(..., alias="sensor_19")
    sensor_20: float = Field(..., alias="sensor_20")
    sensor_21: float = Field(..., alias="sensor_21")

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    """Prediction response."""
    engine_id: int
    cycle: int
    predicted_rul: float = Field(..., description="Predicted Remaining Useful Life in cycles")
    message: str = "Prediction successful"


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    data: List[SensorData] = Field(..., description="List of sensor readings")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total: int


@app.on_event("startup")
async def load_model():
    """Load model and metadata on startup."""
    global model, feature_cols, sensors_to_exclude
    
    model_path = Path('model.pkl')
    if not model_path.exists():
        raise FileNotFoundError(
            "Model file 'model.pkl' not found! Please run train.py first to train a model."
        )
    
    print("Loading model and preprocessing information...")
    model = joblib.load('model.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    sensors_to_exclude = joblib.load('sensors_to_exclude.pkl')
    
    print(f"Model loaded: {type(model).__name__}")
    print(f"Number of features: {len(feature_cols)}")
    if sensors_to_exclude:
        print(f"Excluded sensors: {sensors_to_exclude}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Aircraft Predictive Maintenance API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict-batch",
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
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "num_features": len(feature_cols) if feature_cols else 0
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(sensor_data: SensorData):
    """
    Predict RUL for a single engine cycle.
    
    Args:
        sensor_data: Sensor readings for one engine cycle
        
    Returns:
        PredictionResponse with predicted RUL
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")
    
    try:
        # Convert to DataFrame
        data_dict = sensor_data.dict(by_alias=True)
        df = pd.DataFrame([data_dict])
        
        # Rename columns to match expected format
        column_mapping = {f"sensor_{i:02d}": f"sensor_{i:02d}" for i in range(1, 22)}
        df = df.rename(columns=column_mapping)
        
        # Apply feature engineering
        df_features = create_features(df, sensors_to_exclude=sensors_to_exclude)
        
        # Select features
        X = df_features[feature_cols].copy()
        
        # Handle missing values and infinities
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Ensure all features are present
        missing_features = set(feature_cols) - set(X.columns)
        if missing_features:
            for feat in missing_features:
                X[feat] = 0
        
        X = X[feature_cols]
        
        # Make prediction
        prediction = model.predict(X)[0]
        prediction = max(0, prediction)  # RUL can't be negative
        
        return PredictionResponse(
            engine_id=sensor_data.engine_id,
            cycle=sensor_data.cycle,
            predicted_rul=float(prediction),
            message="Prediction successful"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict RUL for multiple engine cycles.
    
    Args:
        request: BatchPredictionRequest with list of sensor readings
        
    Returns:
        BatchPredictionResponse with list of predictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")
    
    try:
        # Convert to DataFrame
        data_list = [item.dict(by_alias=True) for item in request.data]
        df = pd.DataFrame(data_list)
        
        # Rename columns to match expected format
        column_mapping = {f"sensor_{i:02d}": f"sensor_{i:02d}" for i in range(1, 22)}
        df = df.rename(columns=column_mapping)
        
        # Apply feature engineering
        df_features = create_features(df, sensors_to_exclude=sensors_to_exclude)
        
        # Select features
        X = df_features[feature_cols].copy()
        
        # Handle missing values and infinities
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Ensure all features are present
        missing_features = set(feature_cols) - set(X.columns)
        if missing_features:
            for feat in missing_features:
                X[feat] = 0
        
        X = X[feature_cols]
        
        # Make predictions
        predictions = model.predict(X)
        predictions = np.maximum(predictions, 0)  # RUL can't be negative
        
        # Create response
        prediction_responses = [
            PredictionResponse(
                engine_id=request.data[i].engine_id,
                cycle=request.data[i].cycle,
                predicted_rul=float(predictions[i]),
                message="Prediction successful"
            )
            for i in range(len(predictions))
        ]
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            total=len(prediction_responses)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
