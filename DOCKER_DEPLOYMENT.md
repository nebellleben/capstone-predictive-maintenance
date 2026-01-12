# Docker Deployment Guide

## Overview

This guide covers the Docker deployment of the Aircraft Predictive Maintenance FastAPI service.

✅ **Status**: Docker image built and tested successfully  
✅ **Image**: `aircraft-maintenance-api:simple`  
✅ **Container**: Running on port 8001  
✅ **Tests**: All endpoints working correctly

---

## Quick Start

### Build the Docker Image

```bash
docker build -f Dockerfile.simple -t aircraft-maintenance-api:simple .
```

### Run the Container

```bash
docker run -d --name aircraft-maintenance -p 8001:8000 aircraft-maintenance-api:simple
```

### Test the Deployment

```bash
# Health check
curl http://localhost:8001/health

# Make a prediction
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "engine_id": 1,
    "cycle": 1,
    "sensor_01": -0.0007,
    "sensor_02": -0.0004,
    ...
  }'
```

---

## Docker Image Details

### Base Image
- **OS**: Python 3.11-slim (Debian-based)
- **Size**: Optimized with minimal dependencies

### Included Components
1. **Application**: FastAPI service (`service_test.py` → `service.py`)
2. **Model Files**:
   - `model.pkl` - Trained LightGBM model
   - `feature_columns.pkl` - Feature metadata
   - `sensors_to_exclude.pkl` - Preprocessing metadata
3. **Dependencies**: All Python packages from `requirements.txt`
4. **Health Check**: Automatic health monitoring

### Exposed Ports
- **8000** (container) → Map to any host port (e.g., 8001)

### Health Check Configuration
- **Interval**: 30 seconds
- **Timeout**: 10 seconds
- **Start Period**: 10 seconds
- **Retries**: 3

---

## Container Management

### View Container Status
```bash
docker ps
```

### View Container Logs
```bash
docker logs aircraft-maintenance
docker logs -f aircraft-maintenance  # Follow logs
```

### Stop Container
```bash
docker stop aircraft-maintenance
```

### Restart Container
```bash
docker start aircraft-maintenance
```

### Remove Container
```bash
docker rm -f aircraft-maintenance
```

### Access Container Shell
```bash
docker exec -it aircraft-maintenance /bin/bash
```

---

## API Endpoints

Once the container is running, access these endpoints:

### Root
- **URL**: `GET http://localhost:8001/`
- **Description**: API information

### Health Check
- **URL**: `GET http://localhost:8001/health`
- **Response**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "model_type": "LGBMRegressor",
    "num_features": 20
  }
  ```

### Prediction
- **URL**: `POST http://localhost:8001/predict`
- **Request Body**:
  ```json
  {
    "engine_id": 1,
    "cycle": 1,
    "sensor_01": -0.0007,
    ... (all 21 sensors)
  }
  ```
- **Response**:
  ```json
  {
    "engine_id": 1,
    "cycle": 1,
    "predicted_rul": 121.67,
    "message": "Prediction successful"
  }
  ```

### API Documentation
- **Swagger UI**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`

---

## Cloud Deployment Options

Your Docker image is production-ready and can be deployed to various cloud platforms:

### 1. Hugging Face Spaces (Gradio)

**Requirements**: Valid HF access token

**Steps**:
1. Get token from https://huggingface.co/settings/tokens
2. Update `.hf_credentials`:
   ```bash
   HF_USERNAME=nebeleben
   HF_TOKEN=hf_YourTokenHere
   ```
3. Run deployment script:
   ```bash
   ./deploy_hf_simple.sh
   ```

### 2. Docker Hub

**Push to Docker Hub**:
```bash
# Tag image
docker tag aircraft-maintenance-api:simple yourusername/aircraft-maintenance:v1

# Login to Docker Hub
docker login

# Push image
docker push yourusername/aircraft-maintenance:v1
```

### 3. AWS ECS/Fargate

**Steps**:
1. Push image to AWS ECR
2. Create ECS task definition
3. Create ECS service
4. Configure load balancer (optional)

**Example ECR Push**:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account-id.dkr.ecr.us-east-1.amazonaws.com

docker tag aircraft-maintenance-api:simple your-account-id.dkr.ecr.us-east-1.amazonaws.com/aircraft-maintenance:latest

docker push your-account-id.dkr.ecr.us-east-1.amazonaws.com/aircraft-maintenance:latest
```

### 4. Google Cloud Run

**Deploy**:
```bash
# Tag for Google Container Registry
docker tag aircraft-maintenance-api:simple gcr.io/your-project-id/aircraft-maintenance:latest

# Push to GCR
docker push gcr.io/your-project-id/aircraft-maintenance:latest

# Deploy to Cloud Run
gcloud run deploy aircraft-maintenance \
  --image gcr.io/your-project-id/aircraft-maintenance:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 5. Azure Container Instances

**Deploy**:
```bash
# Login to Azure
az login

# Create resource group
az group create --name aircraft-maintenance-rg --location eastus

# Create container instance
az container create \
  --resource-group aircraft-maintenance-rg \
  --name aircraft-maintenance \
  --image your-registry/aircraft-maintenance:latest \
  --dns-name-label aircraft-maintenance-api \
  --ports 8000
```

### 6. Digital Ocean App Platform

1. Push image to Docker Hub or DigitalOcean Container Registry
2. Create new app in DO dashboard
3. Select "Docker Hub" or "Container Registry"
4. Configure service with your image
5. Set port to 8000
6. Deploy

### 7. Heroku Container Registry

**Deploy**:
```bash
# Login to Heroku
heroku login
heroku container:login

# Create app
heroku create aircraft-maintenance-api

# Push and release
heroku container:push web -a aircraft-maintenance-api
heroku container:release web -a aircraft-maintenance-api
```

### 8. Fly.io

**Deploy**:
```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Launch app
fly launch --image aircraft-maintenance-api:simple

# Deploy
fly deploy
```

### 9. Railway.app

1. Push image to Docker Hub
2. Create new project in Railway
3. Deploy from Docker Hub
4. Configure environment variables (if needed)
5. Deploy

---

## Environment Variables (Optional)

If you need to customize the deployment, you can pass environment variables:

```bash
docker run -d \
  --name aircraft-maintenance \
  -p 8001:8000 \
  -e LOG_LEVEL=debug \
  -e MODEL_PATH=/app/model.pkl \
  aircraft-maintenance-api:simple
```

---

## Troubleshooting

### Container Won't Start

**Check logs**:
```bash
docker logs aircraft-maintenance
```

**Common issues**:
- Model files missing
- Port already in use
- Insufficient memory

### Health Check Failing

**Check health status**:
```bash
docker inspect --format='{{json .State.Health}}' aircraft-maintenance
```

**Test manually**:
```bash
curl http://localhost:8001/health
```

### Predictions Failing

**Check if model loaded**:
```bash
docker logs aircraft-maintenance | grep "Model loaded"
```

**Verify feature columns**:
```bash
docker exec aircraft-maintenance python -c "import joblib; print(joblib.load('feature_columns.pkl'))"
```

### High Memory Usage

**Check container stats**:
```bash
docker stats aircraft-maintenance
```

**Limit memory** (if needed):
```bash
docker run -d \
  --name aircraft-maintenance \
  -p 8001:8000 \
  --memory="512m" \
  aircraft-maintenance-api:simple
```

---

## Performance Tuning

### Workers Configuration

For production, use multiple Uvicorn workers:

**Modify Dockerfile.simple**:
```dockerfile
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Resource Limits

Set CPU and memory limits:
```bash
docker run -d \
  --name aircraft-maintenance \
  -p 8001:8000 \
  --cpus="2" \
  --memory="1g" \
  aircraft-maintenance-api:simple
```

---

## Security Considerations

### 1. Don't Run as Root

**Update Dockerfile**:
```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

### 2. Use Secrets for Credentials

Never hardcode credentials in the image. Use environment variables or secrets management:

```bash
docker run -d \
  --name aircraft-maintenance \
  -p 8001:8000 \
  -e API_KEY=your-secret-key \
  aircraft-maintenance-api:simple
```

### 3. Enable HTTPS

Use a reverse proxy (nginx, Traefik) or cloud platform's HTTPS termination.

### 4. Network Isolation

Use Docker networks for service isolation:
```bash
docker network create aircraft-network
docker run -d --network aircraft-network ...
```

---

## Monitoring and Logging

### Application Logs
```bash
docker logs -f aircraft-maintenance
```

### Health Monitoring
```bash
watch -n 5 'curl -s http://localhost:8001/health | jq'
```

### Resource Monitoring
```bash
docker stats aircraft-maintenance
```

---

## Backup and Versioning

### Save Docker Image
```bash
docker save aircraft-maintenance-api:simple > aircraft-maintenance.tar
```

### Load Docker Image
```bash
docker load < aircraft-maintenance.tar
```

### Tag Versions
```bash
docker tag aircraft-maintenance-api:simple aircraft-maintenance-api:v1.0.0
docker tag aircraft-maintenance-api:simple aircraft-maintenance-api:latest
```

---

## Testing the Deployment

### Automated Test Script

Create `test_docker_deployment.sh`:
```bash
#!/bin/bash
set -e

echo "Testing Docker deployment..."

# Health check
echo "1. Testing health endpoint..."
curl -f http://localhost:8001/health || exit 1
echo " PASSED"

# Prediction test
echo "2. Testing prediction endpoint..."
curl -f -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"engine_id": 1, "cycle": 1, "sensor_01": -0.0007, ...}' \
  || exit 1
echo " PASSED"

echo "All tests passed!"
```

Run:
```bash
chmod +x test_docker_deployment.sh
./test_docker_deployment.sh
```

---

## Next Steps

1. ✅ Docker image built and tested locally
2. 🔲 Choose a cloud platform for deployment
3. 🔲 Set up CI/CD pipeline (optional)
4. 🔲 Configure monitoring and alerting
5. 🔲 Set up custom domain (optional)
6. 🔲 Enable HTTPS
7. 🔲 Implement authentication (if needed)

---

## Support

For issues or questions:
- **GitHub**: https://github.com/nebellleben/capstone-predictive-maintenance
- **Docker Docs**: https://docs.docker.com/

---

**Last Updated**: January 12, 2026  
**Docker Image**: `aircraft-maintenance-api:simple`  
**Status**: ✅ Production Ready
