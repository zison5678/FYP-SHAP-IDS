# FYP-SHAP-IDS

This repository contains the source code for my Final Year Project:

**Enhancing Intrusion Detection Systems Using Explainable Artificial Intelligence with SHAP-Based Machine Learning Models**

## Project Description

This project develops and evaluates a SHAP-guided machine learning framework for multiclass intrusion detection using the CICIDS2017 dataset.

## Main Files

- `main_shap_guided_ids.py`  
  Main implementation for preprocessing, model training, SHAP-guided feature selection, hyperparameter tuning, and evaluation.

- `shap_analysis_and_plots.py`  
  Supporting implementation for SHAP importance analysis and visualisation.

## Dataset

The CICIDS2017 dataset is not included in this repository due to file size limitations. The dataset should be downloaded separately from the official Canadian Institute for Cybersecurity source.

## Models Used

- Logistic Regression
- Linear SVC
- K-Nearest Neighbours
- Random Forest
- Extra Trees

## Feature Selection Strategies

- All Features
- SHAP Global Top-20
- SHAP Classwise Union
- SHAP Hybrid
- SHAP Compact Repeated Priority
