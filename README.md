---
title: Pneumonia Detection API
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: 1.0.0
app_file: src/main.py
pinned: false
---


# Pneumonia Detection API

This is a FastAPI backend for detecting pneumonia from chest X-ray images.  
It loads a pre-trained model (`resnet_pneumonia.h5`) and provides REST API endpoints.


## Project Description
Develop a deep learning model that can detect pneumonia (or other lung diseases) from chest X-ray images. The system should classify whether an image is **Normal** or **Pneumonia** and provide visual explanations for predictions.