import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("best_fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI Enhancements
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="centered")

# Custom CSS for Styling
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #FF1E1E;
    }
    .stTitle {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.title("💳 Fraud Detection System")
st.markdown("### 🔍 Enter Transaction Details (Time, V1 - V5, Amount)")

# Input fields
time = st.number_input("Time", value=0.0)
v1 = st.number_input("V1", value=0.0)
v2 = st.number_input("V2", value=0.0)
v3 = st.number_input("V3", value=0.0)
v4 = st.number_input("V4", value=0.0)
v5 = st.number_input("V5", value=0.0)
amount = st.number_input("Amount", value=0.0)

# Prepare input array
input_array = np.array([[time, v1, v2, v3, v4, v5, amount]])

# Scale the 'Amount' feature
input_array[:, -1] = scaler.transform(input_array[:, -1].reshape(-1, 1))

# Predict button
if st.button("🚀 Check for Fraud"):
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1] * 100  # Fraud Probability

    st.markdown("---")  # Separator line
    
    if prediction == 1:
        st.error(f"🚨 **Fraudulent Transaction Detected!**\n⚠️ Probability: {probability:.2f}%")
    else:
        st.success(f"✅ **Transaction is Legitimate**\n🔵 Probability: {100 - probability:.2f}%")
