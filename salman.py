import streamlit as st       
import numpy as np         
import tensorflow as tf     
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd    
import pickle
import altair as alt
from tensorflow.keras.models import load_model

# Load model and preprocessing tools
model = load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

# Manually define the expected columns used during training
expected_columns = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    # Add your one-hot encoded geography columns exactly as used during training
    'Geography_France', 'Geography_Germany', 'Geography_Spain'
]

# UI
st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('CreditScore')
estimated_salary = st.number_input('EstimatedSalary')
tenure = st.slider('Tenure', 0, 7)
num_of_products = st.slider('NumOfProducts', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Input dataframe
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine and align with expected features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Add missing columns (if any)
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Ensure correct column order
input_data = input_data[expected_columns]

# Predict
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Output
st.write(f"Churn Probability: {prediction_proba:.2f}")
if prediction_proba > 0.5:
    st.write("The Customer is likely to leave the bank")
else:
    st.write("The Customer will not leave the bank")
