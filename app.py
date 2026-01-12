"""
Gradio interface for Aircraft Predictive Maintenance model.
Designed for deployment on Hugging Face Spaces.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

import gradio as gr

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


def load_model():
    """Load model and metadata."""
    global model, feature_cols, sensors_to_exclude
    
    model_path = Path('model.pkl')
    if not model_path.exists():
        # Try alternative paths for Hugging Face Spaces
        model_path = Path('/app/model.pkl')
        if not model_path.exists():
            return False, "Model file not found. Please ensure model.pkl exists."
    
    try:
        model = joblib.load(model_path)
        feature_cols = joblib.load('feature_columns.pkl')
        sensors_to_exclude = joblib.load('sensors_to_exclude.pkl')
        return True, f"Model loaded successfully: {type(model).__name__}"
    except Exception as e:
        return False, f"Error loading model: {str(e)}"


def predict_rul(engine_id, cycle, *sensor_values):
    """
    Predict RUL from sensor inputs.
    
    Args:
        engine_id: Engine unit ID
        cycle: Cycle number
        sensor_values: 21 sensor readings
        
    Returns:
        Predicted RUL in cycles
    """
    if model is None:
        return "Error: Model not loaded. Please check if model.pkl exists."
    
    try:
        # Create DataFrame from inputs
        sensor_dict = {
            'engine_id': [int(engine_id)],
            'cycle': [int(cycle)]
        }
        
        # Add sensor values
        for i, val in enumerate(sensor_values, 1):
            sensor_dict[f'sensor_{i:02d}'] = [float(val)]
        
        df = pd.DataFrame(sensor_dict)
        
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
        
        return f"Predicted RUL: {prediction:.2f} cycles"
    
    except Exception as e:
        return f"Error making prediction: {str(e)}"


# Load model on startup
model_loaded, load_message = load_model()
if not model_loaded:
    print(f"Warning: {load_message}")


# Create Gradio interface
with gr.Blocks(title="Aircraft Predictive Maintenance") as demo:
    gr.Markdown("""
    # Aircraft Engine Predictive Maintenance
    
    Predict the Remaining Useful Life (RUL) of aircraft engines using sensor data.
    
    Enter the engine ID, cycle number, and 21 sensor readings to get a prediction.
    """)
    
    with gr.Row():
        with gr.Column():
            engine_id_input = gr.Number(
                label="Engine ID",
                value=1,
                precision=0
            )
            cycle_input = gr.Number(
                label="Cycle Number",
                value=1,
                precision=0
            )
            
            gr.Markdown("### Sensor Readings")
            
            # Create 21 sensor input fields in a grid
            sensor_inputs = []
            with gr.Row():
                for i in range(1, 8):
                    sensor_inputs.append(gr.Number(
                        label=f"Sensor {i:02d}",
                        value=0.0,
                        precision=6
                    ))
            
            with gr.Row():
                for i in range(8, 15):
                    sensor_inputs.append(gr.Number(
                        label=f"Sensor {i:02d}",
                        value=0.0,
                        precision=6
                    ))
            
            with gr.Row():
                for i in range(15, 22):
                    sensor_inputs.append(gr.Number(
                        label=f"Sensor {i:02d}",
                        value=0.0,
                        precision=6
                    ))
            
            predict_btn = gr.Button("Predict RUL", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="Prediction Result",
                lines=5,
                interactive=False
            )
            
            gr.Markdown(f"""
            ### Model Status
            {load_message}
            """)
    
    # Set up prediction function
    predict_btn.click(
        fn=predict_rul,
        inputs=[engine_id_input, cycle_input] + sensor_inputs,
        outputs=output
    )
    
    gr.Markdown("""
    ### Instructions
    1. Enter the engine ID and current cycle number
    2. Enter all 21 sensor readings
    3. Click "Predict RUL" to get the predicted Remaining Useful Life
    4. The result shows the predicted number of cycles until engine failure
    
    **Note**: Make sure model.pkl, feature_columns.pkl, and sensors_to_exclude.pkl are available.
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
