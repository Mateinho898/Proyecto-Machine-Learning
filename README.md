# GRD Prediction Project - Hospital El Pino

This repository contains the Machine Learning pipeline to predict **Diagnosis-Related Groups (GRD)** for patients at Hospital El Pino.

## Project Structure
- `preprocess.py`: Data cleaning and feature engineering (CIE-10 code extraction).
- `train.py`: Model training and evaluation using Random Forest.
- `requirements.txt`: List of necessary Python libraries.
- `grd_model.pkl`: The final trained model.

## Methodology
The dataset was preprocessed to count the number of comorbidities and procedures per patient. We addressed the extreme class imbalance (500+ categories) by focusing on classes with sufficient data and using **Weighted F1-Score** as the primary performance metric.

## How to use
1. Install dependencies:
   ```bash
   pip install -r requirements.txt