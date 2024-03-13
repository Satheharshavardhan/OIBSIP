import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load the data
@st.cache_data
def load_data():
    spam_data = pd.read_csv("spam.csv", encoding='latin1')
    columns_drop = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]
    spam_data = spam_data.drop(columns_drop, axis=1)
    spam_data.columns = ["Target", "Data"]
    return spam_data

# Data preprocessing
def preprocess_data(spam_data):
    le = LabelEncoder()
    spam_data["Target"] = le.fit_transform(spam_data["Target"])
    feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
    X = feature_extraction.fit_transform(spam_data["Data"])
    Y = spam_data["Target"]
    return X, Y, feature_extraction

# Train the model
def train_model(X, Y):
    model = LogisticRegression()
    model.fit(X, Y)
    return model

# Main function
def main():
    st.title("Spam Classifier")
    st.write("This app classifies text messages as spam or not spam.")

    # Load data
    spam_data = load_data()

    # Preprocess data
    X, Y, feature_extraction = preprocess_data(spam_data)

    # Train model
    model = train_model(X, Y)

    # Text input for user
    user_input = st.text_area("Enter your message here:")

    # Classify button
    if st.button("Classify"):
        if user_input:
            # Feature extraction
            input_features = feature_extraction.transform([user_input])
            # Prediction
            prediction = model.predict(input_features)[0]
            prediction_label = "Spam" if prediction == 1 else "Not Spam"
            st.write("Classification:", prediction_label)

if __name__ == "__main__":
    main()
