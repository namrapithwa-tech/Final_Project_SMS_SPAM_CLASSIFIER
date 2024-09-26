# app.py

import streamlit as st
import pickle
import pandas as pd

# Load the trained model and CountVectorizer
with open("spam_detection_model.pkl", "rb") as model_file:
    model, cv = pickle.load(model_file)

# Title
st.title("SMS Spam Detection App")

# Input text box
user_input = st.text_area("Enter the SMS text:")

# Prediction function
def predict_spam(text):
    # Transform text using the loaded CountVectorizer
    transformed_text = cv.transform([text])
    
    # Predict using the loaded model
    prediction = model.predict(transformed_text)
    
    # Return the result
    return "Spam" if prediction == 1 else "Not Spam"

# Predict button
if st.button("Predict"):
    if user_input:
        result = predict_spam(user_input)
        st.write(f"The message is: *{result}*")
    else:
        st.write("Please enter a message for prediction.")