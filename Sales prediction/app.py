import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Advertising.csv")

# Train model
def train_model(X_train, Y_train):
    model = RandomForestRegressor(n_estimators=99)
    model.fit(X_train, Y_train)
    return model

# Main function
def main():
    st.title("Sales Prediction App")
    st.write("This app predicts sales based on advertising data.")

    # Load data
    data = load_data()
    data1 = data.drop("Unnamed: 0",axis=1)

    # Show raw data
    if st.checkbox("Show raw data"):
        st.write(data1)

    # Data preprocessing
    X = data.drop(["Sales", "Unnamed: 0"], axis=1)  # Remove the "Unnamed: 0" column if it's not needed
    Y = data["Sales"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=7, test_size=0.2)

    # Train model
    model = train_model(X_train, Y_train)

    # Prediction
    st.header("Enter Advertising Budgets")
    tv_budget = st.text_input("TV Advertising Budget", value="100")
    radio_budget = st.text_input("Radio Advertising Budget", value="20")
    newspaper_budget = st.text_input("Newspaper Advertising Budget", value="30")

    # Predict sales
    if st.button("Predict"):
        try:
            tv_budget = float(tv_budget)
            radio_budget = float(radio_budget)
            newspaper_budget = float(newspaper_budget)
            prediction = model.predict([[tv_budget, radio_budget, newspaper_budget]])
            st.success(f"Predicted Sales: {prediction[0]}")
        except ValueError:
            st.error("Please enter valid numerical values.")

if __name__ == "__main__":
    main()
