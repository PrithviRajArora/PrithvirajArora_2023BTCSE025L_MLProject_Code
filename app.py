import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="CloudBurst Predictor", layout="wide")

# Style
st.markdown("""
    <style>
    .stSelectbox label, .stNumberInput label { font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Head
st.title("CloudBurst Prediction System")
st.markdown("Evaluate the probability of a cloudburst (rainfall ≥ 100 mm/h) occurring tomorrow based on current meteorological data.")
st.markdown("---")

# Load Model, Preprocessor & Scaler
@st.cache_resource
def load_components():
    model = joblib.load('cloudburst_rf_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    scaler = joblib.load('scaler.pkl')  # The crucial missing scaler!
    return model, preprocessor, scaler

try:
    rf_model, preprocessor, scaler = load_components()
except FileNotFoundError:
    st.error("Model files not found. Ensure 'cloudburst_rf_model.pkl', 'preprocessor.pkl', and 'scaler.pkl' are in the directory.")
    st.stop()

# True + samples
presets = {
    "Manual Entry (Defaults)": {
        'Location': 'Sydney', 'WindGustDirection': 'W', 'WindDirection9am': 'W', 'WindDirection3pm': 'W',
        'MinimumTemperature': 12.0, 'MaximumTemperature': 24.0, 'Temperature9am': 16.0, 'Temperature3pm': 22.0,
        'Rainfall': 0.0, 'WindGustSpeed': 40.0, 'WindSpeed9am': 15.0, 'WindSpeed3pm': 20.0,
        'Humidity9am': 60.0, 'Humidity3pm': 50.0, 'Pressure9am': 1015.0, 'Pressure3pm': 1012.0,
        'Cloud9am': 4.0, 'Cloud3pm': 4.0
    },
    "[Verified High Risk] Model True Positive 1": {
        'Location': 'Sydney', 'WindGustDirection': 'WNW', 'WindDirection9am': 'NW', 'WindDirection3pm': 'WNW',
        'MinimumTemperature': 3.5, 'MaximumTemperature': 14.4, 'Temperature9am': 5.7, 'Temperature3pm': 13.6,
        'Rainfall': 0.8, 'WindGustSpeed': 43.0, 'WindSpeed9am': 6.0, 'WindSpeed3pm': 22.0,
        'Humidity9am': 100.0, 'Humidity3pm': 76.0, 'Pressure9am': 1015.1, 'Pressure3pm': 1012.6,
        'Cloud9am': 5.0, 'Cloud3pm': 5.0
    },
    "[Verified High Risk] Model True Positive 2": {
        'Location': 'Brisbane', 'WindGustDirection': 'SE', 'WindDirection9am': 'SSE', 'WindDirection3pm': 'SE',
        'MinimumTemperature': 22.4, 'MaximumTemperature': 30.5, 'Temperature9am': 26.2, 'Temperature3pm': 27.2,
        'Rainfall': 0.4, 'WindGustSpeed': 50.0, 'WindSpeed9am': 17.0, 'WindSpeed3pm': 31.0,
        'Humidity9am': 71.0, 'Humidity3pm': 75.0, 'Pressure9am': 1012.0, 'Pressure3pm': 1009.9,
        'Cloud9am': 6.0, 'Cloud3pm': 6.0
    },
    "[Verified High Risk] Model True Positive 3": {
        'Location': 'Darwin', 'WindGustDirection': 'SW', 'WindDirection9am': 'W', 'WindDirection3pm': 'SW',
        'MinimumTemperature': 16.9, 'MaximumTemperature': 23.3, 'Temperature9am': 18.3, 'Temperature3pm': 22.2,
        'Rainfall': 31.8, 'WindGustSpeed': 30.0, 'WindSpeed9am': 20.0, 'WindSpeed3pm': 22.0,
        'Humidity9am': 90.0, 'Humidity3pm': 72.0, 'Pressure9am': 1021.0, 'Pressure3pm': 1021.2,
        'Cloud9am': 8.0, 'Cloud3pm': 8.0
    },
    "[Low Risk] Perfect Sunny Day": {
        'Location': 'Adelaide', 'WindGustDirection': 'NNE', 'WindDirection9am': 'NE', 'WindDirection3pm': 'NNE',
        'MinimumTemperature': 14.0, 'MaximumTemperature': 28.0, 'Temperature9am': 18.0, 'Temperature3pm': 26.0,
        'Rainfall': 0.0, 'WindGustSpeed': 25.0, 'WindSpeed9am': 10.0, 'WindSpeed3pm': 15.0,
        'Humidity9am': 45.0, 'Humidity3pm': 30.0, 'Pressure9am': 1024.0, 'Pressure3pm': 1020.0,
        'Cloud9am': 1.0, 'Cloud3pm': 1.0
    },
    "[Low Risk] Cool Clear Winter": {
        'Location': 'Hobart', 'WindGustDirection': 'S', 'WindDirection9am': 'SSW', 'WindDirection3pm': 'S',
        'MinimumTemperature': 4.0, 'MaximumTemperature': 12.0, 'Temperature9am': 7.0, 'Temperature3pm': 11.0,
        'Rainfall': 0.0, 'WindGustSpeed': 20.0, 'WindSpeed9am': 5.0, 'WindSpeed3pm': 10.0,
        'Humidity9am': 60.0, 'Humidity3pm': 45.0, 'Pressure9am': 1028.0, 'Pressure3pm': 1025.0,
        'Cloud9am': 2.0, 'Cloud3pm': 3.0
    }
}

# PresetS 
selected_preset_name = st.selectbox("Load Sample Data (Optional)", list(presets.keys()))
data = presets[selected_preset_name]

st.markdown("<br>", unsafe_allow_html=True)

# Input Form 
with st.form("prediction_form"):
    st.subheader("Meteorological Readings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Location & Wind Direction**")
        loc_options = ['Albury', 'Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Hobart', 'Darwin', 'Canberra']
        # Fallback in case the location isn't in the list
        loc_index = loc_options.index(data['Location']) if data['Location'] in loc_options else 0
        location = st.selectbox("Location", loc_options, index=loc_index)
        
        wind_dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        wind_gust_dir = st.selectbox("Wind Gust Direction", wind_dirs, index=wind_dirs.index(data['WindGustDirection']))
        wind_dir_9am = st.selectbox("Wind Direction (9 AM)", wind_dirs, index=wind_dirs.index(data['WindDirection9am']))
        wind_dir_3pm = st.selectbox("Wind Direction (3 PM)", wind_dirs, index=wind_dirs.index(data['WindDirection3pm']))

    with col2:
        st.markdown("**Temperature & Rain**")
        min_temp = st.number_input("Minimum Temperature (°C)", value=float(data['MinimumTemperature']))
        max_temp = st.number_input("Maximum Temperature (°C)", value=float(data['MaximumTemperature']))
        temp_9am = st.number_input("Temperature (9 AM) (°C)", value=float(data['Temperature9am']))
        temp_3pm = st.number_input("Temperature (3 PM) (°C)", value=float(data['Temperature3pm']))
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=float(data['Rainfall']))

    with col3:
        st.markdown("**Atmospherics & Wind Speed**")
        wind_gust_speed = st.number_input("Wind Gust Speed (km/h)", min_value=0.0, value=float(data['WindGustSpeed']))
        wind_speed_9am = st.number_input("Wind Speed (9 AM) (km/h)", min_value=0.0, value=float(data['WindSpeed9am']))
        wind_speed_3pm = st.number_input("Wind Speed (3 PM) (km/h)", min_value=0.0, value=float(data['WindSpeed3pm']))
        
        c3_1, c3_2 = st.columns(2)
        with c3_1:
            humidity_9am = st.number_input("Humidity 9AM (%)", min_value=0.0, max_value=100.0, value=float(data['Humidity9am']))
            pressure_9am = st.number_input("Pressure 9AM (hPa)", min_value=900.0, max_value=1050.0, value=float(data['Pressure9am']))
            cloud_9am = st.number_input("Cloud 9AM (oktas)", min_value=0.0, max_value=9.0, value=float(data['Cloud9am']))
        with c3_2:
            humidity_3pm = st.number_input("Humidity 3PM (%)", min_value=0.0, max_value=100.0, value=float(data['Humidity3pm']))
            pressure_3pm = st.number_input("Pressure 3PM (hPa)", min_value=900.0, max_value=1050.0, value=float(data['Pressure3pm']))
            cloud_3pm = st.number_input("Cloud 3PM (oktas)", min_value=0.0, max_value=9.0, value=float(data['Cloud3pm']))

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Execute Prediction")


if submitted:
    user_input_dict = {
        'MinimumTemperature': [min_temp],
        'MaximumTemperature': [max_temp],
        'Rainfall': [rainfall],
        'WindGustSpeed': [wind_gust_speed],
        'WindSpeed9am': [wind_speed_9am],
        'WindSpeed3pm': [wind_speed_3pm],
        'Humidity9am': [humidity_9am],
        'Humidity3pm': [humidity_3pm],
        'Pressure9am': [pressure_9am],
        'Pressure3pm': [pressure_3pm],
        'Cloud9am': [cloud_9am],
        'Cloud3pm': [cloud_3pm],
        'Temperature9am': [temp_9am],
        'Temperature3pm': [temp_3pm],
        'Location': [location],
        'WindGustDirection': [wind_gust_dir],
        'WindDirection9am': [wind_dir_9am],
        'WindDirection3pm': [wind_dir_3pm]
    }
    
    input_df = pd.DataFrame(user_input_dict)
    
    try:
        # Handle missing values and encoding
        processed_data = preprocessor.transform(input_df)
        
        # Scaled the data
        scaled_data = scaler.transform(processed_data)
        
        # Made Prediction using the properly scaled data
        prediction = rf_model.predict(scaled_data)[0]
        probability = rf_model.predict_proba(scaled_data)[0][1]

        st.markdown("---")
        st.subheader("Prediction Output")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if prediction == 1:
                st.error("Status: High Risk. Cloudburst conditions detected.")
            else:
                st.success("Status: Low Risk. No cloudburst expected.")
                
        with res_col2:
            st.metric(label="Calculated Probability", value=f"{probability:.1%}")
            
    except Exception as e:
        st.error(f"Execution error: {e}")