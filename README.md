---
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

## How to Use

1. Enter the **Engine ID** and **Current Cycle Number**
2. Enter all 21 **Sensor Readings**
3. Click **Predict RUL**
4. View the predicted remaining useful life in cycles

## Model Performance

- **Algorithm**: LightGBM
- **MAE**: 13.65 cycles
- **R²**: 0.777
- **Training Data**: 100 engines, 20,631 cycles

## ML Zoomcamp 2025 Project

Capstone project for Machine Learning Zoomcamp by DataTalks.Club.

**GitHub**: [nebellleben/capstone-predictive-maintenance](https://github.com/nebellleben/capstone-predictive-maintenance)
