---
title: TerraVision AI
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.32.0
app_file: streamlit_app.py
pinned: false
---

# TerraVision AI: Multi-Task Precision Agriculture Pipeline

An end-to-end computer vision and precision agriculture prediction system.

## Features
- **Video Preprocessing:** Automated frame extraction with Laplacian blur filtering and LAB color space normalization.
- **Multi-Task Vision Model:** PyTorch CNN predicting 7 Soil Types and 3 Soil Moisture Levels.
- **Context-Aware Recommender:** XGBoost classifier mapping soil and climate attributes to Top-3 Crop Recommendations across 22 classes.
- **Explainable AI:** Grad-CAM visual heatmaps and SHAP feature attribution.
