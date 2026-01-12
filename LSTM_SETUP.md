# LSTM Model Setup

## Issue

TensorFlow/Keras currently doesn't support Python 3.14. The LSTM models in `main.ipynb` require TensorFlow 2.15+.

## Solution

Create a separate Python 3.11 environment for running LSTM models.

### Option 1: Create Python 3.11 environment with uv

```bash
# Install Python 3.11 if not already installed
# macOS:
brew install python@3.11

# Create new environment with Python 3.11
uv venv --python python3.11 .venv-lstm

# Activate the environment
source .venv-lstm/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Option 2: Use conda

```bash
# Create conda environment with Python 3.11
conda create -n predictive-maintenance-lstm python=3.11

# Activate
conda activate predictive-maintenance-lstm

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Skip LSTM for now

The notebook includes traditional ML models (XGBoost, LightGBM) that work with Python 3.14. You can:
1. Run the notebook up to section 6.4 (before LSTM)
2. Compare only traditional ML models
3. Come back to LSTM models later with Python 3.11

## Running LSTM Models

Once you have Python 3.11 environment:

```bash
# Activate Python 3.11 environment
source .venv-lstm/bin/activate  # or conda activate

# Run the notebook or execute specific cells
jupyter notebook main.ipynb

# Or run the full training script
python train.py
```

## Current Status

- **Python 3.14 environment**: Contains all dependencies except TensorFlow/Keras
- **LSTM cells added**: Ready in `main.ipynb` (cells 45-50)
- **Traditional ML models**: Fully functional (Linear, Ridge, Random Forest, XGBoost, LightGBM)

## Recommendation

For ML Zoomcamp submission, you can either:
1. **Submit without LSTM**: Use the best traditional ML model (likely XGBoost or LightGBM)
2. **Create Python 3.11 environment**: Add LSTM comparison to your results

The project is complete and submission-ready with traditional ML models. LSTM is an enhancement.
