
import streamlit as st
import joblib
import numpy as np

# Load model and encoder
regressor = joblib.load("weather_model.pkl")
le = joblib.load("le_weather.pkl")

st.set_page_config(page_title="Weather Prediction App", layout="centered")
st.markdown("# 🌤 Weather Prediction (Decision Tree)")

# ------------------------
# User inputs
# ------------------------
temp = st.number_input("Temperature (°C)", value=25.0)
dew_point = st.number_input("Dew Point Temp (°C)", value=15.0)
rel_humidity = st.number_input("Relative Humidity (%)", value=50.0)
wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
visibility = st.number_input("Visibility (km)", value=10.0)
pressure = st.number_input("Pressure (kPa)", value=101.3)
hour = st.number_input("Hour (0-23)", value=12)
month = st.number_input("Month (1-12)", value=6)

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

    # Display result in a styled circle
    st.markdown(f"""<div style='
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
'>
✅ {weather_pred[0]}
</div>""", unsafe_allow_html=True)
