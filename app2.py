

import streamlit as st  # For creating the web app
import sklearn
import pandas as pd  # For handling data
import pickle  # For loading the trained model
from sklearn.preprocessing import RobustScaler  # For scaling continuous features


# Load the trained model from the pickle file
with open('rm_best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Function to preprocess the data
def preprocess_data(df):
    # Get all numerical features
    num_features = [feature for feature in df.columns if df[feature].dtype != 'O']
    
    # Identify discrete features (features with ≤ 25 unique values)
    discrete_features = [feature for feature in num_features if len(df[feature].unique()) <= 25]
    
    # Get continuous features (numerical features that are not discrete)
    continuous_features = [feature for feature in num_features if feature not in discrete_features]
    
    # Print the list of continuous features to debug
    print(f'Continuous Features: {continuous_features}')
    
    # Apply RobustScaler to continuous features
    scaler = RobustScaler()
    if len(continuous_features) > 0:
        df[continuous_features] = scaler.fit_transform(df[continuous_features])
    else:
        print("No continuous features found for scaling.")
    
    return df

# Streamlit app code
st.title("Heart Attack Prediction")

# User input form
age = st.slider("Age", 29, 77)
sex = st.selectbox("Sex", [0, 1])  # 0 = Female, 1 = Male
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trtbps = st.slider("Resting Blood Pressure (in mm Hg)", 94, 200)
chol = st.slider("Serum Cholestoral (in mg/dl)", 126, 564)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])  # 0 = < 120 mg/dl, 1 = >= 120 mg/dl
restecg = st.selectbox("Resting Electrocardiographic Result", [0, 1, 2])
thalachh = st.slider("Maximum Heart Rate Achieved", 71, 202)
exng = st.selectbox("Exercise Induced Angina", [0, 1])  # 0 = No, 1 = Yes
oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.2)
slp = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
caa = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
thall = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Create a DataFrame from user input
input_data = pd.DataFrame([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]],
                          columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall'])

# Display the input data for debugging
st.write("Input Data for Prediction:")
st.write(input_data)

# Add a button that triggers the prediction
if st.button('Predict Heart Attack Risk'):
    # Preprocess the data
    input_data = preprocess_data(input_data)
    
    # Make predictions using the loaded model
    prediction = best_model.predict(input_data)
    
    # Display the prediction
    if prediction[0] == 1:
        st.write("Prediction: The model predicts that you may have a heart attack.")
    else:
        st.write("Prediction: The model predicts that you are not likely to have a heart attack.")

