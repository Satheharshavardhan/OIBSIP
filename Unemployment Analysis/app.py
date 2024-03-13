import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Load data (replace with your actual data loading logic)
data_upto_nov_2020 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# Feature engineering and normalization (performed outside the app for efficiency)
def preprocess_data(data):
    columns_remove = ['Region', ' Frequency', 'Region.1']
    data = data.drop(columns=columns_remove, axis=1)
    data[" Date"] = pd.to_datetime(data[" Date"])
    data.sort_values(by=" Date", inplace=True)

    today = pd.to_datetime('today').normalize()
    data['Days Since Date'] = (today - data[" Date"]).dt.days
    data.drop(" Date", axis=1, inplace=True)

    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return normalized_data

# Preprocess data (assuming you have the data loaded)
data_upto_nov_2020 = preprocess_data(data_upto_nov_2020)

# Split data into features (X) and target (Y)
X = data_upto_nov_2020.drop(' Estimated Unemployment Rate (%)', axis=1)
Y = data_upto_nov_2020[" Estimated Labour Participation Rate (%)"]

# Preprocess user input for prediction
def preprocess_user_input(data):
    df = pd.DataFrame([data])  # Convert user input to DataFrame
    df = preprocess_data(df)  # Apply same preprocessing steps as for main data
    return df

# Create Streamlit app
st.title("Unemployment Rate Prediction App")

# User input section
st.subheader("Enter feature values:")
user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(col, min_value=X[col].min(), max_value=X[col].max())

# Preprocess user input
user_data = preprocess_user_input(user_input)

# Model loading (assuming the model is saved or trained earlier)
model = SVR()  # Load your trained SVR model here

# Make prediction
if st.button("Predict"):
    prediction = model.predict(user_data)[0]
    st.success(f"Predicted Unemployment Rate: {prediction:.2f}%")

# Display training performance metrics (optional)
# st.subheader("Model Performance (on training data)")
# st.write(f"Mean Squared Error: {mse_train:.2f}")
# st.write(f"R-squared Score: {r2_train:.2f}")
