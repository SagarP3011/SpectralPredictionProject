# SpectralPredictionProject
This repository contains a Machine Learning-based Spectral Data Prediction system, including a Streamlit web application for interactive predictions. The project applies Principal Component Analysis (PCA) for dimensionality reduction and uses a Random Forest Regressor for predictions.

# Project Overview
The goal of this project is to preprocess spectral data, reduce dimensionality using PCA, and make predictions using a trained Random Forest model. Users can upload a CSV file containing spectral data, and the system will:

1.Preprocess the data (remove unnecessary columns, scale features).
2.Apply PCA transformation.
3.Predict the target variable using the Random Forest model.
4.Display the predictions.

# Data Preprocessing
1.Column Removal: The hsi_id column is removed as it is non-numeric.
2.Feature Scaling: Standardization is applied using StandardScaler.
3.Dimensionality Reduction: PCA is applied to reduce high-dimensional spectral data.

# Model Details
Tried Various models and selected the best performing one 
PCA Model (pca_model.pkl): Trained to extract the most significant features.
Random Forest Model (random_forest_model.pkl): Used for regression-based predictions.

# Performance Metrics
The Random Forest model was selected after evaluating multiple models. Key metrics include:

Mean Absolute Error (MAE): Best performance among tested models.
Root Mean Squared Error (RMSE): Indicates the average error in predictions.
R² Score: Measures how well the model explains variability.

# Sample Prediction Flow
1.Upload a CSV file (excluding hsi_id).
2.The app preprocesses and scales the data.
3.PCA transformation is applied to reduce dimensions.
4.Predictions are generated using the trained Random Forest model.
5.The results are displayed in the Streamlit UI.

# Future Improvements
1.Hyperparameter tuning for better model accuracy.
2.More advanced dimensionality reduction techniques (e.g., t-SNE, Autoencoders).
3.Deployment on cloud platforms for public access.
