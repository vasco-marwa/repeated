import numpy as np
import joblib

model = joblib.load("weather_model.pkl")

def predict_weather(temp, humidity, wind, pressure):
    data = np.array([[temp, humidity, wind, pressure]])
    prediction = model.predict(data)
    return prediction[0]
