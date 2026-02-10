import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("Heart Disease Prediction Website 🫀")
st.markdown("Provide the following details")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
restingBP = st.number_input("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fastingBs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restingecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
maxHR = st.slider("Max Heart Rate", 60, 220, 150)
exerciseAngina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_data = {
        'Age': age,
        'RestingBP': restingBP,
        'Cholesterol': cholesterol,
        'FastingBS': fastingBs,
        'MaxHR': maxHR,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + restingecg: 1,
        'ExerciseAngina_' + exerciseAngina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_data])
    
    # Fill missing columns with 0
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match expected format
    input_df = input_df[expected_columns]
    
    # Scale the data
    scaled_data = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(scaled_data)[0]
    
    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("💪 Low Risk of Heart Disease")