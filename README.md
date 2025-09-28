# Hybrid-Deep-Feature-Extraction
Hybrid Deep Feature Extraction and Machine Learning for Brain Tumor Classification: A Comparative Analysis
# Hybrid Deep Feature Extraction for Brain Tumor Classification

This repository contains the implementation of a **Hybrid Deep Feature Extraction** pipeline for brain tumor classification, designed for reproducibility of the experiments presented in our ACIT 2025 paper.

## 📌 Features
- Image preprocessing using **Keras ImageDataGenerator**
- Feature extraction with **ResNet50** (ImageNet pretrained)
- Class balancing using **SMOTE**
- Multiple classifiers:
  - Support Vector Machine (SVM)
  - Random Forest
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
- Hyperparameter tuning with **GridSearchCV**
- Performance evaluation with:
  - Accuracy comparison plots
  - Confusion matrices
 - Automatic export of:
  - `detailed_results.csv`
  - `best_classifier_report.csv`
  - `all_predictions.csv`
  - `results_summary.txt`
- Saves best model (`best_model.pkl`) and feature extractor (`feature_extractor.h5`)

