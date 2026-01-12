#!/bin/bash
# Script to deploy simplified Gradio app to Hugging Face Spaces
# Uses credentials from .hf_credentials file

set -e

echo "=============================================="
echo "  Hugging Face Spaces Deployment Script"
echo "=============================================="
echo ""

# Load credentials from local file (gitignored)
if [ -f .hf_credentials ]; then
    source .hf_credentials
    export HF_USERNAME
    export HF_TOKEN
else
    echo "❌ Error: .hf_credentials file not found!"
    echo "Please create .hf_credentials with your Hugging Face credentials:"
    echo "HF_USERNAME=your_username"
    echo "HF_TOKEN=your_token"
    exit 1
fi

echo "✅ Credentials loaded for user: ${HF_USERNAME}"
echo ""

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "📦 Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Space name
SPACE_NAME="aircraft-predictive-maintenance"
SPACE_ID="${HF_USERNAME}/${SPACE_NAME}"

echo "🚀 Deploying to Hugging Face Space: ${SPACE_ID}"
echo ""

# Check if model files exist
if [ ! -f "model.pkl" ]; then
    echo "❌ Warning: model.pkl not found. Train the model first with: python train.py"
    exit 1
fi

if [ ! -f "feature_columns.pkl" ] || [ ! -f "sensors_to_exclude.pkl" ]; then
    echo "❌ Warning: Metadata files not found"
    exit 1
fi

echo "✅ Model files found"
echo ""

# Login to Hugging Face
echo "🔐 Logging in to Hugging Face..."
python -c "
from huggingface_hub import login
import os
login(token=os.environ.get('HF_TOKEN'))
print('✅ Login successful')
"
echo ""

# Create or get Space
echo "📦 Creating/updating Space..."
python << 'EOPY'
from huggingface_hub import create_repo, HfApi
import os

space_id = os.environ.get('SPACE_ID')
try:
    create_repo(repo_id=space_id, repo_type='space', space_sdk='gradio', exist_ok=True)
    print(f"✅ Space ready: {space_id}")
except Exception as e:
    print(f"Note: {e}")
EOPY

echo ""

# Upload files
echo "📤 Uploading files to Space..."
python << 'EOPY'
from huggingface_hub import HfApi
import os
import shutil

api = HfApi()
space_id = os.environ.get('SPACE_ID')

# Copy app_simple.py to app.py for the Space
shutil.copy('app_simple.py', 'app.py')

# Files to upload
files = [
    'app.py',
    'requirements.txt',
    'model.pkl',
    'feature_columns.pkl',
    'sensors_to_exclude.pkl'
]

# Create README for the Space
readme_content = """---
title: Aircraft Predictive Maintenance
emoji: ✈️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# ✈️ Aircraft Engine Predictive Maintenance

Predict the Remaining Useful Life (RUL) of aircraft engines using machine learning.

## About

This application uses a LightGBM model trained on aircraft engine sensor data to predict how many more operational cycles an engine can safely run before requiring maintenance.

## Features

- **Real-time Predictions**: Enter sensor readings and get instant RUL predictions
- **21 Sensor Inputs**: Comprehensive engine health monitoring
- **Performance**: MAE = 13.65 cycles, R² = 0.777
- **Training Data**: 100 engines, 20,631 operational cycles

## How to Use

1. Enter the **Engine ID** and **Current Cycle Number**
2. Enter all 21 **Sensor Readings** from your engine
3. Click **Predict RUL** to get the prediction
4. The result shows the predicted number of cycles until maintenance is needed

## Model Details

- **Algorithm**: LightGBM Regressor
- **Features**: 20 (15 active sensors + 5 rolling mean features)
- **Preprocessing**: Automatic handling of constant sensors and feature engineering

## ML Zoomcamp 2025 Project

This is a capstone project for the Machine Learning Zoomcamp by DataTalks.Club.

**Source Code**: [GitHub Repository](https://github.com/nebellleben/capstone-predictive-maintenance)

## License

MIT License - Feel free to use and modify!
"""

with open('README.md', 'w') as f:
    f.write(readme_content)

files.append('README.md')

# Upload each file
for file in files:
    if os.path.exists(file):
        print(f"  Uploading {file}...")
        try:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=space_id,
                repo_type='space',
            )
            print(f"    ✅ {file} uploaded")
        except Exception as e:
            print(f"    ⚠️  {file}: {e}")
    else:
        print(f"  ⚠️  {file} not found, skipping")

print("")
print("✅ All files uploaded!")

# Clean up temporary files
if os.path.exists('app.py') and os.path.exists('app_simple.py'):
    if os.path.getmtime('app_simple.py') > os.path.getmtime('app.py'):
        os.remove('app.py')  # Remove the temporary copy

EOPY

echo ""
echo "=============================================="
echo "  ✅ Deployment Complete!"
echo "=============================================="
echo ""
echo "🌐 Your Space is available at:"
echo "   https://huggingface.co/spaces/${SPACE_ID}"
echo ""
echo "Note: It may take a few minutes for the Space to build and start."
echo ""
