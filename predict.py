"""
Prediction script for Aircraft Predictive Maintenance.
Loads trained model and generates RUL predictions for test data.

Note: For web service deployment, use service.py (FastAPI) or app.py (Gradio).
This script is for batch file-based predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

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


def load_test_data(data_dir='dataset'):
    """Load test data."""
    print("Loading test data...")
    data_dir = Path(data_dir)
    
    # Column names
    column_names = ['engine_id', 'cycle'] + [f'sensor_{i:02d}' for i in range(1, 22)]
    
    # Load test data
    # Note: Using sep=r'\s+' to handle multiple spaces correctly
    test_df = pd.read_csv(
        data_dir / 'PM_test.txt',
        sep=r'\s+',
        header=None,
        names=column_names,
        usecols=range(23)  # Only use first 23 columns
    )
    
    print(f"Test data shape: {test_df.shape}")
    return test_df


def prepare_test_features(test_df, sensors_to_exclude, feature_cols):
    """Prepare test features using the same feature engineering as training."""
    print("Creating features for test data...")
    
    # Apply feature engineering
    test_features = create_features(test_df, sensors_to_exclude=sensors_to_exclude)
    
    # Select features (same as training)
    X_test = test_features[feature_cols].copy()
    
    # Handle any issues
    X_test = X_test.fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    
    # Ensure all training features are present
    missing_features = set(feature_cols) - set(X_test.columns)
    if missing_features:
        print(f"Warning: Missing features in test data: {missing_features}")
        for feat in missing_features:
            X_test[feat] = 0
    
    # Select only the features that were used in training
    X_test = X_test[feature_cols]
    
    print(f"Test feature matrix shape: {X_test.shape}")
    return X_test, test_features


def predict_rul_per_engine(test_df, test_features, predictions):
    """
    For each engine, predict RUL at the last cycle.
    The test data contains multiple cycles per engine, but we need one RUL per engine.
    """
    # Get the last cycle for each engine
    last_cycles = test_df.groupby('engine_id')['cycle'].max().reset_index()
    last_cycles.columns = ['engine_id', 'last_cycle']
    
    # Create a mapping from (engine_id, cycle) to prediction
    test_features_with_pred = test_features[['engine_id', 'cycle']].copy()
    test_features_with_pred['predicted_rul'] = predictions
    
    # Merge with last cycles to get predictions at last cycle for each engine
    engine_rul = test_features_with_pred.merge(
        last_cycles, 
        left_on=['engine_id', 'cycle'], 
        right_on=['engine_id', 'last_cycle'],
        how='inner'
    )
    
    # Sort by engine_id to match ground truth order
    engine_rul = engine_rul.sort_values('engine_id')
    
    return engine_rul['predicted_rul'].values


def main():
    """Main prediction function."""
    print("="*60)
    print("Aircraft Predictive Maintenance - Prediction")
    print("="*60)
    
    # Check if model exists
    if not Path('model.pkl').exists():
        print("Error: Model file 'model.pkl' not found!")
        print("Please run train.py first to train a model.")
        return
    
    # Load model and metadata
    print("\nLoading trained model and preprocessing information...")
    model = joblib.load('model.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    sensors_to_exclude = joblib.load('sensors_to_exclude.pkl')
    
    print(f"Model loaded: {type(model).__name__}")
    print(f"Number of features: {len(feature_cols)}")
    if sensors_to_exclude:
        print(f"Excluded sensors: {sensors_to_exclude}")
    
    # Load test data
    test_df = load_test_data()
    
    # Prepare features
    X_test, test_features = prepare_test_features(test_df, sensors_to_exclude, feature_cols)
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = model.predict(X_test)
    
    # Ensure predictions are non-negative (RUL can't be negative)
    predictions = np.maximum(predictions, 0)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction range: {predictions.min():.2f} to {predictions.max():.2f}")
    print(f"Mean prediction: {predictions.mean():.2f}")
    
    # Get RUL per engine (last cycle of each engine)
    engine_rul_predictions = predict_rul_per_engine(test_df, test_features, predictions)
    
    print(f"\nEngine-level predictions shape: {engine_rul_predictions.shape}")
    print(f"Engine-level prediction range: {engine_rul_predictions.min():.2f} to {engine_rul_predictions.max():.2f}")
    
    # Save predictions
    output_file = 'predictions.txt'
    print(f"\nSaving predictions to {output_file}...")
    np.savetxt(output_file, engine_rul_predictions, fmt='%.2f')
    
    # Also save as CSV with engine IDs for reference
    unique_engines = sorted(test_df['engine_id'].unique())
    predictions_df = pd.DataFrame({
        'engine_id': unique_engines,
        'predicted_rul': engine_rul_predictions
    })
    predictions_df.to_csv('predictions.csv', index=False)
    
    # If ground truth is available, calculate metrics
    truth_file = Path('dataset/PM_truth.txt')
    if truth_file.exists():
        print("\nEvaluating predictions against ground truth...")
        truth = pd.read_csv(truth_file, header=None, names=['RUL'])
        truth = truth['RUL'].values
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(truth, engine_rul_predictions)
        rmse = np.sqrt(mean_squared_error(truth, engine_rul_predictions))
        r2 = r2_score(truth, engine_rul_predictions)
        
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Save evaluation metrics
        metrics_df = pd.DataFrame({
            'metric': ['MAE', 'RMSE', 'R2'],
            'value': [mae, rmse, r2]
        })
        metrics_df.to_csv('evaluation_metrics.csv', index=False)
        print("Evaluation metrics saved to 'evaluation_metrics.csv'")
    
    print(f"\nPredictions saved to '{output_file}' and 'predictions.csv'")
    print("Prediction complete!")


if __name__ == '__main__':
    main()
