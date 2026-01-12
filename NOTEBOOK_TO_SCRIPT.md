# Notebook-to-Script Workflow

## Overview

This project follows a **notebook-first approach** where `main.ipynb` is the single source of truth, and `train.py` is automatically generated from it.

## Workflow

```
main.ipynb (Source of Truth)
     ↓ (jupyter nbconvert)
train.py (Generated Script)
```

## Files

### main.ipynb
- **Purpose**: Complete ML pipeline with EDA, feature engineering, and training
- **Contains**: All code, visualizations, and documentation
- **Status**: Manually edited and maintained

### train.py
- **Purpose**: Executable training script for production/automation
- **Contains**: Same code as notebook, without visualization cells
- **Status**: Auto-generated from notebook

## Making Changes

### When you edit the notebook:

1. **Edit `main.ipynb`** with your changes
2. **Regenerate `train.py`**:
   ```bash
   jupyter nbconvert --to script main.ipynb --output train
   ```
3. **Test the script**:
   ```bash
   python train.py
   ```

### DO NOT directly edit train.py

- train.py will be overwritten when regenerated
- All changes should be made in main.ipynb

## What was unified?

The following sections were added to `main.ipynb`:

1. **Model Training** (Section 6)
   - Train/validation split (engine-based)
   - Evaluation function

2. **Baseline Models** (Section 6.1)
   - Linear Regression
   - Ridge Regression
   - Random Forest

3. **Hyperparameter Optimization** (Section 6.2)
   - XGBoost optimization with Optuna
   - LightGBM optimization with Optuna

4. **Model Comparison and Selection** (Section 6.3)
   - Compare all models
   - Select best model based on validation MAE

5. **Model Persistence** (Section 6.4)
   - Save best model
   - Save feature metadata
   - Save results summary

## Script Cleanup

The generated script is automatically cleaned:
- Removed cell markers (In[ ]:, In[1]:, etc.)
- Removed markdown comments
- Added proper docstring header
- Cleaned up formatting

## Benefits

1. **Single Source of Truth**: All code maintained in one place
2. **Consistency**: Script always matches notebook
3. **Documentation**: Notebook serves as both code and documentation
4. **Reproducibility**: Easy to understand entire pipeline
5. **Best Practice**: Follows Jupyter → Script workflow

## Quick Reference

### Generate train.py from notebook
```bash
jupyter nbconvert --to script main.ipynb --output train
```

### Run the generated script
```bash
python train.py
```

### Verify script syntax
```bash
python -m py_compile train.py
```
