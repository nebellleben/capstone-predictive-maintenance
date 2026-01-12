# Aircraft Predictive Maintenance

Capstone project for Machine Learning Zoomcamp Cohort 2025

## Problem Description

Aircraft engine failure is a critical safety and operational concern in aviation. Unplanned engine failures can lead to:
- **Safety risks**: Engine failures during flight pose serious safety hazards
- **Operational disruptions**: Unexpected maintenance causes flight delays and cancellations
- **Economic losses**: Emergency repairs and aircraft downtime result in significant costs
- **Resource inefficiency**: Reactive maintenance is less efficient than planned maintenance

### The Solution

This project develops a machine learning model to predict the **Remaining Useful Life (RUL)** of aircraft engines using sensor data. By accurately forecasting when an engine will fail, airlines can:

- **Schedule proactive maintenance** during planned downtime, reducing unplanned failures
- **Optimize maintenance costs** by performing maintenance only when needed
- **Improve safety** by identifying engines at risk before critical failures occur
- **Increase operational efficiency** by minimizing aircraft downtime

### How the Model Works

The model uses time-series sensor data from multiple aircraft engines to learn degradation patterns. Each engine has 21 sensors that monitor various operational parameters (temperature, pressure, vibration, etc.). As engines age and degrade, these sensor readings change in predictable ways. The model:

1. **Analyzes sensor trends** over time to identify degradation patterns
2. **Engineers features** from raw sensor data (rolling statistics, rate of change, aggregations)
3. **Predicts RUL** in cycles (operational cycles until failure)
4. **Provides actionable insights** for maintenance scheduling

**Problem Type**: Time-series regression  
**Input**: Sensor readings from multiple aircraft engines over time (21 sensors per cycle)  
**Output**: Remaining Useful Life (RUL) in cycles

## Dataset

The dataset is from [Kaggle - Predictive Maintenance Data](https://www.kaggle.com/datasets/maternusherold/pred-maintanance-data?resource=download)

### Data Structure

- **PM_train.txt**: Training data containing time-series sensor readings
  - Format: `engine_id cycle sensor_1 sensor_2 ... sensor_21`
  - Each row represents one cycle of one engine
  - 21 sensor measurements per cycle
  - Multiple engines with varying lifespans
  
- **PM_test.txt**: Test data with the same format (no labels)

- **PM_truth.txt**: Ground truth RUL values for test engines
  - One RUL value per engine (the remaining cycles until failure)

### Data Format

Each line in the training/test files contains:
- Engine unit ID (integer)
- Cycle number (integer)
- 21 sensor measurements (floats)

### Downloading the Dataset

The dataset should be placed in the `dataset/` directory. If you need to download it:

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/maternusherold/pred-maintanance-data?resource=download)
2. Download the files: `PM_train.txt`, `PM_test.txt`, and `PM_truth.txt`
3. Place them in the `dataset/` folder in this project

## Project Structure

```
├── main.ipynb          # EDA, data exploration, and analysis
├── train.py            # Model training script
├── predict.py          # Batch prediction script for test data
├── service.py          # FastAPI web service for model predictions
├── app.py              # Gradio interface for Hugging Face deployment
├── requirements.txt    # Python dependencies (pip)
├── pyproject.toml      # Python dependencies (uv)
├── Dockerfile          # Container setup for Jupyter notebook
├── Dockerfile.service  # Container setup for web service
├── README.md           # This file
└── dataset/            # Data files
    ├── PM_train.txt
    ├── PM_test.txt
    └── PM_truth.txt
```

## Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installing uv

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or using pip:
```bash
pip install uv
```

### Setting Up the Environment

#### Using uv (Recommended)

1. **Create and activate virtual environment:**
```bash
# Create virtual environment
uv venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

2. **Install dependencies:**
```bash
uv pip install -r requirements.txt
```

Or using pyproject.toml:
```bash
uv pip install -e .
```

#### Using pip

1. **Create virtual environment:**
```bash
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### macOS-Specific Setup

**Important for macOS users**: XGBoost and LightGBM require the OpenMP runtime library. Install it using Homebrew:

```bash
brew install libomp
```

After installation, you may need to set the library path (add this to your `~/.zshrc` or `~/.bash_profile`):

```bash
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
```

Then reload your shell configuration:
```bash
source ~/.zshrc  # or source ~/.bash_profile
```

## Usage

### 1. Exploratory Data Analysis

Open `main.ipynb` in Jupyter to explore the data, perform EDA, and understand the dataset characteristics.

```bash
jupyter notebook main.ipynb
```

### 2. Train Model

Train the model using the training script:

```bash
python train.py
```

**Note**: `train.py` is automatically generated from `main.ipynb` using `jupyter nbconvert`. The notebook is the single source of truth for all code.

This will:
- Load and preprocess the training data
- Perform feature engineering
- Train multiple models (Linear Regression, Ridge, Random Forest, XGBoost, LightGBM)
- Perform hyperparameter tuning using Optuna
- Select the best model based on validation MAE
- Save the trained model and preprocessing pipeline to:
  - `model.pkl` - Trained model
  - `feature_columns.pkl` - Feature column names
  - `sensors_to_exclude.pkl` - Excluded sensors
  - `model_results.csv` - Model comparison results

#### Regenerating train.py from the Notebook

If you make changes to `main.ipynb` and want to update `train.py`:

```bash
jupyter nbconvert --to script main.ipynb --output train
```

This ensures the training script always matches the notebook.

### 3. Make Batch Predictions

For batch predictions on test data files:

```bash
python predict.py
```

This will:
- Load the trained model
- Process test data from `dataset/PM_test.txt`
- Generate RUL predictions
- Save predictions to `predictions.txt` and `predictions.csv`
- If ground truth is available, calculate and save evaluation metrics

### 4. Web Service (FastAPI)

Start the FastAPI web service for interactive predictions:

```bash
python service.py
```

Or using uvicorn directly:
```bash
uvicorn service:app --host 0.0.0.0 --port 8000
```

The service will be available at:
- **API**: http://localhost:8000
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

#### API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `POST /predict` - Single prediction endpoint
  ```json
  {
    "engine_id": 1,
    "cycle": 50,
    "sensor_01": -0.0007,
    "sensor_02": -0.0004,
    ...
    "sensor_21": 23.4190
  }
  ```
- `POST /predict-batch` - Batch prediction endpoint
  ```json
  {
    "data": [
      {
        "engine_id": 1,
        "cycle": 50,
        "sensor_01": -0.0007,
        ...
      },
      ...
    ]
  }
  ```

#### Example API Usage

Using curl:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "engine_id": 1,
    "cycle": 50,
    "sensor_01": -0.0007,
    "sensor_02": -0.0004,
    "sensor_03": 100.0,
    "sensor_04": 518.67,
    "sensor_05": 641.82,
    "sensor_06": 1589.70,
    "sensor_07": 1400.60,
    "sensor_08": 14.62,
    "sensor_09": 21.61,
    "sensor_10": 554.36,
    "sensor_11": 2388.06,
    "sensor_12": 9046.19,
    "sensor_13": 1.30,
    "sensor_14": 47.47,
    "sensor_15": 521.66,
    "sensor_16": 2388.02,
    "sensor_17": 8138.62,
    "sensor_18": 8.4195,
    "sensor_19": 0.03,
    "sensor_20": 392,
    "sensor_21": 2388
  }'
```

Using Python:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "engine_id": 1,
        "cycle": 50,
        "sensor_01": -0.0007,
        # ... all 21 sensors
    }
)
print(response.json())
```

### 5. Gradio Interface (Local)

Run the Gradio interface locally:

```bash
python app.py
```

The interface will be available at http://localhost:7860

## Deployment

### Docker Deployment

#### Web Service Container

Build and run the FastAPI service:

```bash
# Build the image
docker build -f Dockerfile.service -t predictive-maintenance-service .

# Run the container
docker run -p 8000:8000 -v $(pwd):/app predictive-maintenance-service
```

The service will be available at http://localhost:8000

#### Jupyter Notebook Container

For running the notebook:

```bash
# Build the image
docker build -t predictive-maintenance .

# Run the container
docker run -p 8888:8888 -v $(pwd):/app predictive-maintenance
```

Access Jupyter at http://localhost:8888

### Hugging Face Spaces Deployment

Deploy the Gradio interface to Hugging Face Spaces. See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

#### Quick Start (Using Deployment Script)

1. **Train the model** (if not already done):
   ```bash
   python train.py
   ```

2. **Run the deployment script**:
   ```bash
   ./deploy_hf.sh
   ```

   The script uses credentials from `.hf_credentials` (gitignored, stored locally).

3. **Access your deployed app**:
   Your app will be available at: `https://huggingface.co/spaces/YOUR_USERNAME/aircraft-predictive-maintenance`

#### Manual Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for manual deployment methods via web interface or Git.

**Note**: 
- Your Hugging Face credentials are stored locally in `.hf_credentials` (gitignored)
- Never commit credentials to the repository
- Make sure model files (`model.pkl`, `feature_columns.pkl`, `sensors_to_exclude.pkl`) exist before deploying

## Methodology

1. **Data Preparation**: Load and clean sensor data, handle missing values, identify constant sensors
2. **Feature Engineering**: Create time-based features, rolling statistics, degradation indicators, engine-level aggregations, sensor interactions
3. **Exploratory Data Analysis**: Understand sensor patterns, correlations, and degradation trends
4. **Model Selection**: Compare baseline models (Linear Regression, Ridge) and advanced models (Random Forest, XGBoost, LightGBM)
5. **Hyperparameter Tuning**: Optimize model parameters using Optuna with TPE sampler
6. **Evaluation**: Use MAE, RMSE, and R² metrics, with focus on early prediction accuracy
7. **RUL Capping**: Apply piecewise linear degradation model (cap RUL at 125 cycles)

### Code Organization

This project follows a **notebook-first approach**:
- **`main.ipynb`**: Complete pipeline with EDA, feature engineering, model training, and evaluation (single source of truth)
- **`train.py`**: Auto-generated from `main.ipynb` using `jupyter nbconvert --to script`
- **`predict.py`**: Batch prediction script for file-based processing
- **`service.py`**: FastAPI web service for real-time predictions
- **`app.py`**: Gradio interface for interactive predictions

To update `train.py` after modifying the notebook:
```bash
jupyter nbconvert --to script main.ipynb --output train
```

## Key Features

- **Time-series data handling** with proper temporal ordering and engine-based train/validation split
- **Piecewise linear RUL degradation modeling** (cap at 125 cycles)
- **Comprehensive feature engineering** for sensor data (rolling stats, degradation indicators, aggregations)
- **Multiple model comparison** and selection (baseline + tree-based models)
- **Hyperparameter optimization** using Optuna
- **REST API** for model serving (FastAPI)
- **Interactive interface** for predictions (Gradio)
- **Containerization** with Docker
- **Cloud deployment** ready for Hugging Face Spaces

## Model Performance

After training, model performance metrics will be saved to `model_results.csv`. The best model is selected based on validation MAE (Mean Absolute Error).

Example metrics:
- **MAE**: Mean Absolute Error in cycles
- **RMSE**: Root Mean Squared Error in cycles
- **R²**: Coefficient of determination

## Results

(Results will be documented after model training. Run `python train.py` to generate model results.)

## Troubleshooting

### Model not found error

If you see "Model file 'model.pkl' not found":
1. Make sure you've run `python train.py` first
2. Check that `model.pkl`, `feature_columns.pkl`, and `sensors_to_exclude.pkl` exist in the project root

### Port already in use

If port 8000 or 7860 is already in use:
- Change the port in `service.py` or `app.py`
- Or stop the process using the port

### Dependency installation issues

If you encounter issues installing dependencies:
- Make sure you're using Python 3.11 or higher
- Try using `uv` instead of `pip` for faster and more reliable installation
- On some systems, you may need to install system dependencies (e.g., `build-essential` on Linux)

## License

This project is part of the Machine Learning Zoomcamp 2025 capstone project.

## References

- [ML Zoomcamp Project Requirements](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/projects/README.md)
- [Kaggle Dataset](https://www.kaggle.com/datasets/maternusherold/pred-maintanance-data?resource=download)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Gradio Documentation](https://www.gradio.app/)
- [Hugging Face Spaces](https://huggingface.co/spaces)
