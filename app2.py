import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('rm_best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Streamlit app
st.title("Heart Attack Prediction")

# User input
age = st.slider("Age", 29, 77)
sex = st.selectbox("Sex", [0, 1])  # 0 = Female, 1 = Male
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trtbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200)
chol = st.slider("Serum Cholesterol (mg/dl)", 126, 564)
fbs = st.selectbox("Fasting Blood Sugar >= 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalachh = st.slider("Max Heart Rate Achieved", 71, 202)
exng = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("ST Depression by Exercise", 0.0, 6.2)
slp = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
caa = st.selectbox("Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
thall = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Convert input to DataFrame
input_data = pd.DataFrame([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]],
                          columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall'])

# Predict
if st.button('Predict'):
    prediction = best_model.predict(input_data)
    result = "High Risk of Heart Attack" if prediction[0] == 1 else "Low Risk of Heart Attack"
    st.write(f"Prediction: {result}")
