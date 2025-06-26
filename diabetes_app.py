import streamlit as st
import base64
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Function to encode image to base64
def set_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set your background image
set_background("main.jpg")

# Load model and scaler from .pkl
with open("diabetes_model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

# Load dataset (for correlation & sample view)
df = pd.read_csv("diabetes.csv")

# App UI
st.set_page_config(page_title="🩺 Diabetes Prediction App", layout="centered")
st.title("🌟 Diabetes Prediction App")
st.write("This app uses ML to predict whether a person is diabetic.")

st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        background-image: url('https://img.freepik.com/premium-photo/diabetes-medical-background-with-symbols-doctors-tools-flat-design-vector-illustration_522823-3058.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧪 Diabetes Risk Predictor")
st.markdown("Enter your health information below to check your diabetes risk.")

# Input sliders
pregnancies = st.slider("Pregnancies", 0, 17, 1)
glucose = st.slider("Glucose Level", 0, 200, 100)
bp = st.slider("Blood Pressure", 0, 122, 70)
skin = st.slider("Skin Thickness", 0, 99, 20)
insulin = st.slider("Insulin Level", 0, 846, 80)
bmi = st.slider("BMI", 0.0, 67.0, 25.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 10, 100, 25)

# Prepare input for prediction
input_data = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
    columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
)
input_scaled = scaler.transform(input_data)

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 High Risk: You are likely Diabetic.\n🧬 Risk Probability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk: You are likely NOT Diabetic.\n🧬 Risk Probability: {probability:.2f}")

# Optional: Show correlation heatmap
with st.expander("📊 Show Feature Correlation Heatmap"):
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Optional: Data summary
with st.expander("📁 Show Sample Data"):
    st.dataframe(df.head())

