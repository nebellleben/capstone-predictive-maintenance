# Testing Summary

## Date: January 12, 2026

## Overview
Comprehensive testing of the Aircraft Predictive Maintenance notebook (`main.ipynb`) and associated scripts.

## Issues Found and Fixed

### 1. Critical Data Loading Bug ❌ → ✅

**Issue**: The original data loading code used `sep=' '` which incorrectly parsed the data files:
- Only loaded 1 engine instead of 100 engines
- Misaligned columns causing incorrect data mapping
- Training data appeared to have only 1 unique engine ID

**Root Cause**: The data files use multiple spaces as separators, which `sep=' '` cannot handle correctly. The files have 26 columns, but only 23 are needed.

**Fix Applied**:
```python
# Before (INCORRECT)
train_df = pd.read_csv(
    DATA_DIR / 'PM_train.txt',
    sep=' ',
    header=None,
    names=column_names,
    engine='python'
)

# After (CORRECT)
train_df = pd.read_csv(
    DATA_DIR / 'PM_train.txt',
    sep=r'\s+',          # Handle multiple spaces
    header=None,
    names=column_names,
    usecols=range(23)    # Only use first 23 columns
)
```

**Files Updated**:
- `main.ipynb` (Cell 3 and Cell 4)
- `train.py` (regenerated from notebook)
- `predict.py` (data loading section)

**Verification**:
- ✅ Training data: 20,631 samples, 100 engines (was 1 engine)
- ✅ Test data: 13,096 samples, 100 engines
- ✅ Engine ID range: 1 to 100
- ✅ Cycle range: 1 to 362
- ✅ No missing values

### 2. XGBoost/LightGBM Missing Dependency ❌ → ✅

**Issue**: XGBoost failed to load with error:
```
Library not loaded: @rpath/libomp.dylib
```

**Root Cause**: OpenMP runtime library not installed on macOS.

**Fix Applied**:
```bash
brew install libomp
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
```

**Verification**:
- ✅ XGBoost imports successfully
- ✅ LightGBM imports successfully
- ✅ All tree-based models train without errors

### 3. TensorFlow/Keras Compatibility ⚠️

**Issue**: TensorFlow/Keras not compatible with Python 3.14.

**Status**: EXPECTED - documented in `LSTM_SETUP.md`

**Solution**: Use Python 3.11 or 3.12 for LSTM models. Traditional ML models work fine with Python 3.14.

## Test Results

### Package Import Test
All required packages successfully imported:
- ✅ pandas, numpy, scikit-learn
- ✅ matplotlib, seaborn
- ✅ xgboost, lightgbm
- ✅ optuna, joblib, scipy
- ✅ fastapi, uvicorn, gradio
- ⚠️ tensorflow, keras (requires Python 3.11/3.12)

### Data Pipeline Test
All data processing steps verified:

1. **Data Loading**:
   - ✅ Training: 20,631 samples, 100 engines
   - ✅ Test: 13,096 samples, 100 engines
   - ✅ Ground truth: 100 RUL values

2. **Data Preprocessing**:
   - ✅ Constant sensor identification: 6 sensors
   - ✅ RUL calculation: range 0-361
   - ✅ RUL capping at 125: range 0-125

3. **Feature Engineering**:
   - ✅ Active sensors: 15 (after removing 6 constant)
   - ✅ Rolling features generated
   - ✅ Time-based features calculated

4. **Train/Validation Split**:
   - ✅ Engine-based split: 80 train / 20 validation engines
   - ✅ Train: 16,340 samples
   - ✅ Validation: 4,291 samples

### Model Training Test
All models trained successfully with reasonable performance:

| Model             | MAE   | RMSE  | R²    | Status |
|-------------------|-------|-------|-------|--------|
| Linear Regression | 17.69 | 21.31 | 0.736 | ✅     |
| XGBoost           | 13.68 | 19.63 | 0.776 | ✅     |
| LightGBM          | 13.65 | 19.58 | 0.777 | ✅     |

**Best Model**: LightGBM with MAE=13.65 cycles

### Performance Metrics Validation
The metrics are reasonable for RUL prediction:
- MAE of ~13-18 cycles is acceptable for engines with max life of 125+ cycles
- R² values of 0.73-0.78 indicate good predictive power
- Tree-based models (XGBoost, LightGBM) outperform linear models as expected

## Environment Configuration

### Working Configuration
- **OS**: macOS (darwin 24.6.0)
- **Python**: 3.14 (for traditional ML models)
- **Virtual Environment**: uv
- **Required System Package**: libomp (via Homebrew)

### Environment Variable
For XGBoost/LightGBM to work correctly:
```bash
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
```

*Note: This should be added to your shell profile for persistence.*

### LSTM Models
For LSTM/TensorFlow support:
- Use Python 3.11 or 3.12
- See `LSTM_SETUP.md` for detailed instructions

## Conclusion

✅ **Notebook Status**: Fully functional

✅ **Critical Issues**: All resolved

⚠️ **Known Limitations**: 
- LSTM models require Python 3.11/3.12 (documented)
- OpenMP library required for XGBoost/LightGBM on macOS

## Recommendations

1. **Document the libomp requirement** in README.md for macOS users
2. **Update Dockerfile** to ensure libomp is available
3. **Add data validation tests** to catch loading issues early
4. **Consider adding unit tests** for critical functions

## Files Modified
- `main.ipynb` - Fixed data loading (cells 3, 4)
- `train.py` - Regenerated from notebook
- `predict.py` - Fixed data loading
- `TESTING_SUMMARY.md` - Created (this file)

## Next Steps
1. Update README.md with macOS setup instructions
2. Test the FastAPI web service (`service.py`)
3. Test the Gradio app (`app.py`)
4. Test Docker containerization
5. Test Hugging Face deployment
