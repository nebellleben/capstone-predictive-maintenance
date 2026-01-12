"""
Simplified Gradio interface for Aircraft Predictive Maintenance model.
Uses the simplified feature engineering matching the trained model.
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


def load_model():
    """Load model and metadata."""
    global model, feature_cols, sensors_to_exclude
    
    model_path = Path('model.pkl')
    if not model_path.exists():
        return False, "Model file not found. Please ensure model.pkl exists."
    
    try:
        model = joblib.load(model_path)
        feature_cols = joblib.load('feature_columns.pkl')
        sensors_to_exclude = joblib.load('sensors_to_exclude.pkl')
        return True, f"✅ Model loaded: {type(model).__name__} with {len(feature_cols)} features"
    except Exception as e:
        return False, f"❌ Error loading model: {str(e)}"


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
        return "❌ Error: Model not loaded. Please check if model.pkl exists."
    
    try:
        # Create DataFrame from inputs
        sensor_dict = {
            'engine_id': [int(engine_id)],
            'cycle': [int(cycle)]
        }
        
        # Add sensor values
        for i, val in enumerate(sensor_values, 1):
            sensor_dict[f'sensor_{i:02d}'] = [float(val) if val != "" else 0.0]
        
        df = pd.DataFrame(sensor_dict)
        
        # Apply simplified feature engineering
        df_features = create_simple_features(df, sensors_to_exclude=sensors_to_exclude)
        
        # Select features
        X = df_features[feature_cols].fillna(0) if all(col in df_features.columns for col in feature_cols) else df_features[feature_cols].reindex(columns=feature_cols, fill_value=0)
        
        # Make prediction
        prediction = model.predict(X)[0]
        prediction = max(0, prediction)  # RUL can't be negative
        
        result = f"""
### 🎯 Prediction Result

**Predicted RUL**: {prediction:.2f} cycles

**Engine ID**: {engine_id}  
**Current Cycle**: {cycle}

---

**Interpretation**:
- The engine is predicted to have approximately **{prediction:.0f} cycles** of useful life remaining.
- This means the engine can operate for about {prediction:.0f} more cycles before maintenance is needed.
- For safety, plan maintenance when RUL reaches 10-20 cycles.
"""
        return result
    
    except Exception as e:
        return f"❌ Error making prediction: {str(e)}"


# Load model on startup
model_loaded, load_message = load_model()
if not model_loaded:
    print(f"Warning: {load_message}")


# Create Gradio interface with improved UI
with gr.Blocks(title="Aircraft Predictive Maintenance", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # ✈️ Aircraft Engine Predictive Maintenance
    
    ### Predict the Remaining Useful Life (RUL) of aircraft engines using sensor data
    
    This ML model predicts how many more operational cycles an aircraft engine can safely run before requiring maintenance.
    Enter the engine ID, cycle number, and 21 sensor readings to get a prediction.
    """)
    
    gr.Markdown(f"""
    ### 📊 Model Status
    {load_message}
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 Engine Information")
            engine_id_input = gr.Number(
                label="Engine ID",
                value=1,
                precision=0,
                info="Unique identifier for the engine"
            )
            cycle_input = gr.Number(
                label="Current Cycle Number",
                value=1,
                precision=0,
                info="Current operational cycle"
            )
            
            gr.Markdown("### 📈 Sensor Readings (1-7)")
            sensor_inputs = []
            with gr.Row():
                for i in range(1, 4):
                    sensor_inputs.append(gr.Number(
                        label=f"Sensor {i:02d}",
                        value=-0.0007 if i == 1 else (-0.0004 if i == 2 else 100.0),
                        precision=6
                    ))
            with gr.Row():
                for i in range(4, 8):
                    sensor_inputs.append(gr.Number(
                        label=f"Sensor {i:02d}",
                        value=518.67 if i == 4 else (641.82 if i == 5 else (1589.7 if i == 6 else 1400.6)),
                        precision=6
                    ))
            
            gr.Markdown("### 📈 Sensor Readings (8-14)")
            with gr.Row():
                for i in range(8, 11):
                    sensor_inputs.append(gr.Number(
                        label=f"Sensor {i:02d}",
                        value=14.62 if i == 8 else (21.61 if i == 9 else 554.36),
                        precision=6
                    ))
            with gr.Row():
                for i in range(11, 15):
                    sensor_inputs.append(gr.Number(
                        label=f"Sensor {i:02d}",
                        value=2388.06 if i == 11 else (9046.19 if i == 12 else (1.30 if i == 13 else 47.47)),
                        precision=6
                    ))
            
            gr.Markdown("### 📈 Sensor Readings (15-21)")
            with gr.Row():
                for i in range(15, 18):
                    sensor_inputs.append(gr.Number(
                        label=f"Sensor {i:02d}",
                        value=521.66 if i == 15 else (2388.02 if i == 16 else 8138.62),
                        precision=6
                    ))
            with gr.Row():
                for i in range(18, 22):
                    sensor_inputs.append(gr.Number(
                        label=f"Sensor {i:02d}",
                        value=8.4195 if i == 18 else (0.03 if i == 19 else (392.0 if i == 20 else 2388.0)),
                        precision=6
                    ))
            
            predict_btn = gr.Button("🚀 Predict RUL", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output = gr.Markdown(
                label="Prediction Result",
                value="Click 'Predict RUL' to see results..."
            )
            
            gr.Markdown("""
            ### 📖 Instructions
            
            1. **Enter Engine ID**: Unique identifier for the engine being analyzed
            2. **Enter Cycle Number**: Current operational cycle
            3. **Enter Sensor Readings**: All 21 sensor values from the engine
            4. **Click Predict**: Get the predicted Remaining Useful Life
            
            ### ℹ️ About RUL
            
            **RUL (Remaining Useful Life)** represents the number of operational cycles 
            the engine can safely run before maintenance is required.
            
            **Example**:
            - RUL = 120 cycles → Engine is healthy, far from failure
            - RUL = 50 cycles → Plan maintenance soon
            - RUL = 10 cycles → Urgent maintenance needed
            
            ### 🔬 Model Details
            
            - **Algorithm**: LightGBM Regressor
            - **Features**: 20 (15 sensors + 5 rolling features)
            - **Performance**: MAE = 13.65 cycles, R² = 0.777
            - **Training Data**: 100 engines, 20,631 samples
            
            ### 🎓 ML Zoomcamp 2025 Project
            
            This is a capstone project for the Machine Learning Zoomcamp.
            """)
    
    # Set up prediction function
    predict_btn.click(
        fn=predict_rul,
        inputs=[engine_id_input, cycle_input] + sensor_inputs,
        outputs=output
    )
    
    gr.Markdown("""
    ---
    
    **Note**: Default sensor values are pre-filled from a sample engine at cycle 1.
    You can modify them to test different scenarios.
    
    **Source**: [GitHub Repository](https://github.com/nebellleben/capstone-predictive-maintenance)
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
