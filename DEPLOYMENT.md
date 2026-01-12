# Deployment Guide

## Hugging Face Spaces Deployment

This guide explains how to deploy the Gradio app to Hugging Face Spaces.

### Prerequisites

1. **Hugging Face Account**: You need a Hugging Face account. If you don't have one, create it at https://huggingface.co/join

2. **Local Credentials**: Your Hugging Face credentials are stored locally in `.hf_credentials` (this file is gitignored and will NOT be committed to the repository).

### Method 1: Using the Deployment Script (Recommended)

1. **Ensure credentials are set up**:
   The `.hf_credentials` file should already exist with your credentials. If not, create it:
   ```bash
   cat > .hf_credentials << EOF
   HF_USERNAME=your_username
   HF_PASSWORD=your_password
   EOF
   ```

2. **Train the model** (if not already done):
   ```bash
   python train.py
   ```
   This will create `model.pkl`, `feature_columns.pkl`, and `sensors_to_exclude.pkl`.

3. **Run the deployment script**:
   ```bash
   ./deploy_hf.sh
   ```

   The script will:
   - Load your credentials from `.hf_credentials`
   - Log in to Hugging Face
   - Create or update your Space
   - Upload all necessary files
   - Deploy the app

4. **Access your deployed app**:
   Your app will be available at: `https://huggingface.co/spaces/YOUR_USERNAME/aircraft-predictive-maintenance`

### Method 2: Manual Deployment via Web Interface

1. **Go to Hugging Face Spaces**: https://huggingface.co/spaces

2. **Create a new Space**:
   - Click "Create new Space"
   - Name: `aircraft-predictive-maintenance`
   - SDK: Select "Gradio"
   - License: Choose appropriate license
   - Click "Create Space"

3. **Upload files**:
   - Upload `app.py`
   - Upload `requirements.txt`
   - Upload `model.pkl` (after training)
   - Upload `feature_columns.pkl`
   - Upload `sensors_to_exclude.pkl`

4. **Wait for build**:
   - Hugging Face will automatically build and deploy your app
   - Check the "Logs" tab for any errors

### Method 3: Manual Deployment via Git

1. **Clone your Space repository**:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/aircraft-predictive-maintenance
   cd aircraft-predictive-maintenance
   ```

2. **Copy files**:
   ```bash
   cp ../app.py .
   cp ../requirements.txt .
   cp ../model.pkl .
   cp ../feature_columns.pkl .
   cp ../sensors_to_exclude.pkl .
   ```

3. **Create README.md for the Space**:
   ```markdown
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
   ```

4. **Commit and push**:
   ```bash
   git add .
   git commit -m "Add Gradio app and model files"
   git push
   ```

### Authentication

For Git-based deployment, you'll need to authenticate:

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Login (you'll be prompted for your token)
huggingface-cli login
```

Or use your password as a token:
```bash
python -c "from huggingface_hub import login; login(token='YOUR_PASSWORD')"
```

### Troubleshooting

1. **Model files not found**:
   - Make sure you've run `python train.py` first
   - Check that `model.pkl`, `feature_columns.pkl`, and `sensors_to_exclude.pkl` exist

2. **Authentication errors**:
   - Verify your credentials in `.hf_credentials`
   - Make sure the file is not committed to git (it's in `.gitignore`)

3. **Build errors**:
   - Check the Space logs in the Hugging Face web interface
   - Verify all dependencies are in `requirements.txt`
   - Make sure `app.py` is in the root of the Space

4. **App not loading**:
   - Check that model files are uploaded
   - Verify the model files are not too large (Hugging Face has size limits)
   - Check the logs for error messages

### Security Notes

- **NEVER commit credentials to git**: The `.hf_credentials` file is gitignored
- **Use environment variables in CI/CD**: For automated deployments, use environment variables instead of credential files
- **Rotate credentials**: If credentials are exposed, change them immediately in your Hugging Face account settings

### Space Configuration

Your Space will be configured with:
- **SDK**: Gradio
- **Python version**: 3.11 (as specified in requirements)
- **Hardware**: CPU (free tier)
- **Auto-refresh**: Enabled (app updates automatically on git push)

### Updating the Deployment

To update your deployed app:

1. **Make changes** to `app.py` or model files locally
2. **Re-run the deployment script**:
   ```bash
   ./deploy_hf.sh
   ```

Or manually:
```bash
cd aircraft-predictive-maintenance
# Make changes
git add .
git commit -m "Update app"
git push
```

The Space will automatically rebuild and update.
