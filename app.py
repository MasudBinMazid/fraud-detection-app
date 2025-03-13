import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("best_fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Fraud Detection System")
st.markdown("### 🔍 Enter Transaction Details (Time, V1 - V28, Amount)")

# Collect all 30 inputs
time = st.number_input("Time", value=0.0)
features = [st.number_input(f"V{i}", value=0.0) for i in range(1, 29)]
amount = st.number_input("Amount", value=0.0)

# Prepare input array (30 features)
input_array = np.array([[time] + features + [amount]])

# Scale amount feature
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
