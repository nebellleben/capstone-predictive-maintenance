# Model Comparison Summary

## Overview

This project implements and compares multiple machine learning approaches for aircraft engine RUL prediction:

### Traditional ML Models
1. **Linear Regression** - Simple baseline
2. **Ridge Regression** - Regularized linear model
3. **Random Forest** - Ensemble of decision trees
4. **XGBoost** - Gradient boosting (optimized with Optuna)
5. **LightGBM** - Gradient boosting (optimized with Optuna)

### Deep Learning Models
6. **Simple LSTM** - Single LSTM layer with dropout
7. **Stacked LSTM** - Multi-layer LSTM with dropout

## Implementation Details

### Traditional ML Models
- **Data**: Tabular features (engineered from sensor data)
- **Features**: 21 sensors + rolling statistics + degradation indicators + aggregations
- **Split**: Engine-based 80/20 split
- **Optimization**: Optuna (30 trials) for XGBoost and LightGBM
- **Training**: Fitted on flattened time-series data

### LSTM Models
- **Data**: Sequential (sequences of 10 cycles)
- **Features**: Normalized sensor values + engineered features
- **Architecture**: 
  - Simple: 50 LSTM units → Dense(25) → Output
  - Stacked: 64 LSTM → 32 LSTM → Dense(16) → Output
- **Regularization**: Dropout (0.2-0.3)
- **Training**: Early stopping, learning rate reduction

## Expected Results

### Typical Performance (based on similar datasets)

**Traditional ML Models:**
- Linear/Ridge: MAE ~15-20 cycles
- Random Forest: MAE ~12-15 cycles
- XGBoost: MAE ~10-13 cycles (likely best)
- LightGBM: MAE ~10-13 cycles

**LSTM Models:**
- Simple LSTM: MAE ~11-14 cycles
- Stacked LSTM: MAE ~10-13 cycles
- May overfit if not enough data

## Running the Models

### Full comparison (requires Python 3.11 for LSTM):
```bash
# Create Python 3.11 environment
uv venv --python python3.11 .venv-lstm
source .venv-lstm/bin/activate
uv pip install -r requirements.txt

# Run notebook
jupyter notebook main.ipynb
```

### Traditional ML only (works with current Python 3.14):
Run the notebook up to Section 6.4 (Model Persistence) to train and compare:
- Linear Regression
- Ridge Regression  
- Random Forest
- XGBoost (optimized)
- LightGBM (optimized)

## Results Visualization

The notebook generates:
1. **Model comparison table** - All metrics side by side
2. **Performance bar charts** - MAE, RMSE, R² comparisons
3. **Training history plots** - For LSTM models
4. **Best model selection** - Automatic based on validation MAE

## Model Selection Criteria

Best model is chosen based on:
1. **Primary**: Lowest validation MAE
2. **Secondary**: Low overfitting (train-val gap)
3. **Practical**: Training time, inference speed

## Recommendations

### For Production:
- **XGBoost/LightGBM**: Best balance of accuracy and speed
- **Random Forest**: More interpretable, robust to hyperparameters
- **LSTM**: If you need to capture complex temporal patterns

### For Submission:
- Use best traditional ML model (XGBoost or LightGBM)
- LSTM is optional enhancement
- Focus on feature engineering quality

## Next Steps

1. Run `python train.py` to train all traditional models
2. (Optional) Set up Python 3.11 for LSTM comparison
3. Review `model_results_comprehensive.csv` for full comparison
4. Deploy best model using `service.py` or `app.py`
