import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load PCA and Random Forest model
@st.cache_resource
def load_pickle_file(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

pca = load_pickle_file("pca_model.pkl")
model = load_pickle_file("random_forest_model.pkl")

st.title("Spectral Data Prediction App")
st.write("Upload a CSV file containing spectral data for preprocessing and prediction.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview:")
        st.write(data.head())

        
        if "hsi_id" in data.columns:
            data = data.drop(columns=["hsi_id"])
            st.info(" `hsi_id` column removed.")

        
        data = data.apply(pd.to_numeric, errors="coerce")

        # Check for NaN values
        if data.isnull().sum().sum() > 0:
            st.error(" Error: The uploaded data contains non-numeric values or missing data. Please check your file.")
        else:
            # Standardize the data (using the same columns as in training)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data.iloc[:, :-1])  # Assuming last column is the target (remove if needed)
            st.info(" Data standardized successfully.")

            transformed_data = pca.transform(scaled_data)
            st.info(f" PCA transformation applied. Reduced to {transformed_data.shape[1]} features.")

            # Make Predictions
            if st.button("Predict"):
                predictions = model.predict(transformed_data)
                st.write("### Predictions:")
                st.write(pd.DataFrame({"Prediction": predictions}))

    except Exception as e:
        st.error(f" An error occurred: {str(e)}")
