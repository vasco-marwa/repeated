import streamlit as st
import numpy as np
import joblib

# -------------------------------------------------
# Load trained model and LabelEncoder (from notebook)
# -------------------------------------------------
model = joblib.load("weather_model.pkl")
le = joblib.load("le_weather.pkl")

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Weather Temperature Prediction",
    page_icon="🌡️",
    layout="centered"
)

# -------------------------------------------------
# Custom CSS (Attractive UI)
# -------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #74ebd5, #9face6);
}
.prediction-circle {
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: linear-gradient(to right, #ff9966, #ff5e62);
    color: white;
    font-size: 32px;
    font-weight: bold;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🌡️ Temperature Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Inputs based on trained notebook model</p>", unsafe_allow_html=True)

# -------------------------------------------------
# Webpage Inputs (MATCH NOTEBOOK FEATURES)
# -------------------------------------------------
dew_point = st.number_input("🌫️ Dew Point Temperature (°C)", -50.0, 50.0, 10.0)
humidity = st.number_input("💧 Relative Humidity (%)", 0.0, 100.0, 60.0)
wind_speed = st.number_input("🌬️ Wind Speed (km/h)", 0.0, 100.0, 10.0)
visibility = st.number_input("👁️ Visibility (km)", 0.0, 50.0, 20.0)
pressure = st.number_input("📈 Pressure (kPa)", 90.0, 110.0, 101.0)

weather_text = st.selectbox(
    "☁️ Weather Condition",
    le.classes_
)

# -------------------------------------------------
# CONNECT WEB INPUTS → TRAINED MODEL
# -------------------------------------------------
if st.button("🔮 Predict Temperature"):
    weather_encoded = le.transform([weather_text])[0]

    X = np.array([[
        dew_point,
        humidity,
        wind_speed,
        visibility,
        pressure,
        weather_encoded
    ]])

    prediction = model.predict(X)[0]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='prediction-circle'>{prediction:.2f} °C</div>",
        unsafe_allow_html=True
    )
