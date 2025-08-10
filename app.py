# app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import datetime

# 💼 App Title
st.title("💼 Salary Prediction App")
st.markdown("Predict your salary based on experience. (Education and Skill are for UI only)")

# 📥 User Inputs
experience = st.number_input("📊 Years of Experience", min_value=0.0, max_value=50.0, step=0.1, format="%.1f")
education = st.selectbox("🎓 Education Level", ["Matric", "Intermediate", "Bachelor's", "Master's", "PhD"])
skill = st.selectbox("🛠️ Skill Level", ["Beginner", "Intermediate", "Advanced"])

# 🧠 Load the model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# 🔮 Predict Salary
if st.button("Predict Salary"):
    scaled_input = scaler.transform([[experience]])
    prediction_usd = model.predict(scaled_input)[0]
    
    # Convert to PKR (you can update exchange rate here)
    conversion_rate = 290
    prediction_pkr = prediction_usd * conversion_rate

    # 💰 Show Prediction
    st.subheader(f"💰 Estimated Salary (USD): ${prediction_usd:,.2f}")
    st.subheader(f"🇵🇰 Estimated Salary (PKR): Rs. {prediction_pkr:,.0f}")

    # 📝 Log (optional)
    with open("logs.txt", "a") as log_file:
        log_file.write(f"[{datetime.datetime.now()}] Exp: {experience}, Edu: {education}, Skill: {skill}, Salary(USD): ${prediction_usd:,.2f}, Salary(PKR): Rs.{prediction_pkr:,.0f}\n")

# 📈 Salary Trend Visualization
st.subheader("📊 Salary Trend Based on Experience")
try:
    df = pd.read_csv("Salary_Data.csv")
    df = df.sort_values("YearsExperience")

    fig, ax = plt.subplots()
    ax.plot(df["YearsExperience"], df["Salary"], marker='o', color='green')
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.set_title("Salary Trend Line")
    st.pyplot(fig)

except Exception as e:
    st.error("Trend graph failed to load.")
    st.text(str(e))
