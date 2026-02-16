import streamlit as st
import joblib
import numpy as np

# ------------------------
# Load trained model & LabelEncoder
# ------------------------
regressor = joblib.load("weather_model.pkl")
le = joblib.load("le_weather.pkl")

# ------------------------
# Page config
# ------------------------
st.set_page_config(page_title="Weather Prediction App", layout="centered")

# ------------------------
# App title and description
# ------------------------
st.markdown("""
# 🌤 Weather Prediction (Decision Tree)
Enter the values below to predict the weather:
""")

# ------------------------
# User inputs
# ------------------------
col1, col2 = st.columns(2)

with col1:
    temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, step=0.1)
    dew_point = st.number_input("Dew Point Temp (°C)", min_value=-50.0, max_value=60.0, step=0.1)
    rel_humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=200.0, step=0.1)
    
with col2:
    visibility = st.number_input("Visibility (km)", min_value=0.0, max_value=100.0, step=0.1)
    pressure = st.number_input("Pressure (kPa)", min_value=80.0, max_value=120.0, step=0.1)
    hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, step=1)
    month = st.number_input("Month (1-12)", min_value=1, max_value=12, step=1)

# ------------------------
# Predict button
# ------------------------
if st.button("Predict Weather"):
    temp_dew_diff = temp - dew_point
    input_features = np.array([[temp, dew_point, rel_humidity, wind_speed,
                                visibility, pressure, hour, month, temp_dew_diff]])
    
    # Predict encoded weather
    weather_encoded = regressor.predict(input_features)
    
    # Decode to original label
    weather_pred = le.inverse_transform(weather_encoded.astype(int))
    
    # Display result in a colored box
    st.markdown(f"""
    <div style="
        text-align:center;
        background: linear-gradient(to right, #27ae60, #2ecc71);
        padding: 40px;
        border-radius: 50%;
        color: white;
        font-size: 24px;
        font-weight: bold;
        width: 220px;
        height: 220px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: auto;
        box-shadow: 0 0 20px #2ecc71;
    ">
        ✅ {weather_pred[0]}
    </div>
    """, unsafe_allow_html=True)
