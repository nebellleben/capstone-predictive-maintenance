#!/bin/bash
# Script to deploy Gradio app to Hugging Face Spaces
# Uses credentials from .hf_credentials file

set -e

# Load credentials from local file (gitignored)
if [ -f .hf_credentials ]; then
    source .hf_credentials
    export HF_USERNAME
    export HF_PASSWORD
else
    echo "Error: .hf_credentials file not found!"
    echo "Please create .hf_credentials with your Hugging Face credentials:"
    echo "HF_USERNAME=your_username"
    echo "HF_PASSWORD=your_password"
    exit 1
fi

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Space name (you can change this)
SPACE_NAME="aircraft-predictive-maintenance"
SPACE_ID="${HF_USERNAME}/${SPACE_NAME}"

echo "Deploying to Hugging Face Space: ${SPACE_ID}"

# Login to Hugging Face
echo "Logging in to Hugging Face..."
python -c "
from huggingface_hub import login
import os
login(token=os.environ.get('HF_PASSWORD'), add_to_git_credential=True)
"

# Check if model files exist
if [ ! -f "model.pkl" ]; then
    echo "Warning: model.pkl not found. Make sure to train the model first with: python train.py"
    exit 1
fi

# Create temporary directory for Space files
TEMP_DIR=$(mktemp -d)
echo "Creating Space files in ${TEMP_DIR}"

# Copy necessary files
cp app.py "${TEMP_DIR}/"
cp requirements.txt "${TEMP_DIR}/"
cp model.pkl "${TEMP_DIR}/" 2>/dev/null || echo "Warning: model.pkl not found"
cp feature_columns.pkl "${TEMP_DIR}/" 2>/dev/null || echo "Warning: feature_columns.pkl not found"
cp sensors_to_exclude.pkl "${TEMP_DIR}/" 2>/dev/null || echo "Warning: sensors_to_exclude.pkl not found"

# Create README for the Space
cat > "${TEMP_DIR}/README.md" << EOF
---
title: Aircraft Predictive Maintenance
emoji: ✈️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Aircraft Predictive Maintenance

Predict the Remaining Useful Life (RUL) of aircraft engines using sensor data.

Enter engine ID, cycle number, and 21 sensor readings to get a prediction.
EOF

# Clone or create the Space repository
if [ -d "${SPACE_NAME}" ]; then
    echo "Space directory exists, updating..."
    cd "${SPACE_NAME}"
    git pull
else
    echo "Creating new Space repository..."
    git clone "https://${HF_USERNAME}:${HF_PASSWORD}@huggingface.co/spaces/${SPACE_ID}" "${SPACE_NAME}" || {
        echo "Space doesn't exist yet. Creating it..."
        python -c "
from huggingface_hub import create_repo
create_repo(repo_id='${SPACE_ID}', repo_type='space', space_sdk='gradio')
"
        git clone "https://${HF_USERNAME}:${HF_PASSWORD}@huggingface.co/spaces/${SPACE_ID}" "${SPACE_NAME}"
    }
    cd "${SPACE_NAME}"
fi

# Copy files to Space directory
cp "${TEMP_DIR}"/* .

# Commit and push
git add .
git commit -m "Update Gradio app and model files" || echo "No changes to commit"
git push

echo "Deployment complete!"
echo "Your Space is available at: https://huggingface.co/spaces/${SPACE_ID}"

# Cleanup
cd ..
rm -rf "${TEMP_DIR}"
