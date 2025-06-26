import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("main.jpg")

# Load the trained model and scaler
with open("diabetes_model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

st.title("Diabetes Prediction App")
st.write("Please enter the details below:")

# User inputs via sliders
pregnancies = st.slider("Pregnancies", 0, 20, 1)
glucose = st.slider("Glucose", 0, 200, 110)
blood_pressure = st.slider("Blood Pressure", 0, 150, 70)
skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin", 0, 900, 80)
bmi = st.slider("BMI", 0.0, 70.0, 25.0)
diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.slider("Age", 10, 100, 30)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1] * 100

    if prediction == 1:
        st.error(f"❌ You may have diabetes with a {proba:.2f}% chance.")
    else:
        st.success(f"✅ You are unlikely to have diabetes. Probability: {proba:.2f}%")

    # Export prediction as PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Diabetes Prediction Result", ln=True, align='C')
    pdf.ln(10)
    result_text = f"Prediction: {'Positive' if prediction else 'Negative'}\nProbability: {proba:.2f}%\n"
    for name, val in zip([
        "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
        "Insulin", "BMI", "Diabetes Pedigree Function", "Age"],
        input_data[0]):
        result_text += f"{name}: {val}\n"
    for line in result_text.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    st.download_button("Download Result as PDF", pdf_output.getvalue(), file_name="diabetes_result.pdf")

# Optional expanders
with st.expander("📊 View Correlation Heatmap"):
    df = pd.read_csv("diabetes.csv")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with st.expander("📁 View Sample Data"):
    st.dataframe(df.head())
