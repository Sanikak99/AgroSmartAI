import streamlit as st
import numpy as np
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")



# Load models and preprocessing objects
xgb_model = joblib.load("xgb_production_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
numeric_feature_names = joblib.load("numeric_feature_names.pkl")

rf_model = joblib.load("rf_fertilizer_model.pkl")
scaler_fert = joblib.load("scaler_fert.pkl")
label_encoder_crop = joblib.load("label_encoder_crop.pkl")
label_encoder_fert = joblib.load("label_encoder_fert.pkl")

# Weather API Key
API_KEY = 'be4c9c6ec928f120c525e614398ffdc2'

# Function to fetch weather
def fetch_weather(city_name):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url).json()
        temp = response['main']['temp']
        humidity = response['main']['humidity']
        wind_speed = response['wind']['speed'] * 3.6
        return temp, humidity, wind_speed
    except:
        return None, None, None

# Streamlit App
st.set_page_config(page_title="AgroSmartAI", layout="wide")
st.title("🌾 AgroSmartAI - Crop Production & Fertilizer Predictor")
tabs = st.tabs(["📈 Crop Production Prediction", "🧪 Fertilizer Recommendation"])

# ------------------
# Crop Production Prediction
# ------------------
with tabs[0]:
    st.header("📈 Predict Crop Production")
    col1, col2 = st.columns(2)

    with col1:
        State = st.selectbox("State", [state.split("_")[-1] for state in label_encoders['Crop_State'].classes_])
        city_name = st.text_input("City Name for Weather", "Amaravati")
        Area = st.number_input("Area (hectares)", min_value=0.0)
        Rain_Annual = st.number_input("🌧️ Annual Rainfall (mm)", min_value=0.0)

    Temperature, Humidity, Wind_Speed = fetch_weather(city_name)

    with col2:
        st.subheader("🌦️ Weather Overview")
        if Temperature is None:
            st.info("Manual Weather Input")
            Temperature = st.slider("🌡️ Temperature (°C)", -10, 60, 25)
            Humidity = st.slider("💧 Humidity (%)", 0, 100, 60)
            Wind_Speed = st.slider("🍃 Wind Speed (km/h)", 0, 200, 10)
        else:
            st.metric("🌡️ Temperature", f"{Temperature} °C")
            st.metric("💧 Humidity", f"{Humidity} %")
            st.metric("🍃 Wind Speed", f"{Wind_Speed:.2f} km/h")

    
    Crop = st.selectbox("Crop", sorted(set(c.split('_')[0] for c in label_encoders['Crop_Season'].classes_)))
    Season = st.selectbox("Season", ['Kharif', 'Rabi', 'Whole Year', 'Winter'])
    

    if st.button("📈 Predict Production"):
        crop_season = f"{Crop}_{Season}"
        crop_state = f"{Crop}_{State}"

        try:
            Crop_Season_enc = label_encoders['Crop_Season'].transform([crop_season])[0]
            Crop_State_enc = label_encoders['Crop_State'].transform([crop_state])[0]

            features = {
                'Area': Area,
                'Temperature': Temperature,
                'Humidity': Humidity,
                'Wind_Speed': Wind_Speed,
                'Area_Temp': Area * Temperature,
                'Temp_Humidity': Temperature * Humidity,
                'Humidity_Wind': Humidity * Wind_Speed,
                'Temp_Wind': Temperature * Wind_Speed,
                'Area_Humidity': Area * Humidity,
                'Area_Temp_Wind': Area * Temperature * Wind_Speed,
                'Humidity_to_Temp': Humidity / (Temperature + 1e-5),
                'Wind_to_Humidity': Wind_Speed / (Humidity + 1e-5),
                'Temperature_Sq': Temperature ** 2,
                'Humidity_Sq': Humidity ** 2,
                'Wind_Speed_Sq': Wind_Speed ** 2,
                'Rainfall_Per_Acre': Rain_Annual / (Area + 1e-5),
                'Temp_Rainfall_Interaction': Temperature * Rain_Annual,
                'Log_Area': np.log1p(Area),
                'Crop_Season': Crop_Season_enc,
                'Crop_State': Crop_State_enc
            }

            input_df = pd.DataFrame([features])[numeric_feature_names]
            log_prediction = xgb_model.predict(input_df)[0]
            production_prediction = np.expm1(log_prediction)
            
            
            yield_per_hectare = production_prediction / (Area + 1e-5)

           
            st.markdown(f"""
            <div style='margin: 40px 0 30px 0; background-color:#1e1e1e;padding:20px 30px;border-radius:12px;
                        border-left:6px solid #00bcd4;box-shadow:0 0 10px rgba(0, 188, 212, 0.4);'>
                <h3 style='color:#00e5ff;'>🌾 Predicted Crop Production: <strong>{production_prediction:,.2f} tons</strong></h3>
                <p style='font-size:18px;color:#80deea;'>📊 Estimated Yield: <strong>{yield_per_hectare:,.2f} tons/hectare</strong></p>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("📉 Risk Indicator")
            if production_prediction < 1:
                st.error("⚠️ Very Low Production Expected - High Risk")
            elif production_prediction < 5:
                st.warning("⚠️ Moderate Risk of Low Production")
            else:
                st.success("✅ Low Risk - Favorable Conditions")

        except:
            st.warning("✨ Let's try that again! Some input values might be missing or out of range. Please review and submit once more — we're almost there! 😊")


# -----------------------------
# Fertilizer Recommendation
# -----------------------------
with tabs[1]:
    st.header("🧪 Recommend Fertilizer")

    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("Select Crop", label_encoder_crop.classes_)
        city_name_fert = st.text_input("City Name for Weather", "Pune", key="fert_weather")
        rain = st.number_input("🌧️ Rainfall (mm)", min_value=0.0)
        carbon = st.number_input("🌱 Carbon Level", min_value=0.0, step=0.1)
    temp, humidity, wind = fetch_weather(city_name_fert)

    with col2:
        st.subheader("🌦️ Weather Overview")
        if temp is None:
            st.info("Manual Weather Input")
            temp = st.slider("🌡️ Temperature (°C)", 0, 60, 25)
            humidity = st.slider("💧 Humidity (%)", 0, 100, 50)
            moist = st.slider("💦 Moisture (%)", 0, 100, 50)
        else:
            st.metric("🌡️ Temperature", f"{temp} °C")
            st.metric("💧 Humidity", f"{humidity} %")
            st.metric("🍃 Wind Speed", f"{wind:.2f} km/h")
            moist = humidity

    
    ph = st.slider("🧪 Soil pH", 0.0, 14.0)
    nitrogen = st.number_input("🌿 Nitrogen Level", min_value=0 )
    phosphorous = st.number_input("🪴 Phosphorous Level", min_value=0)
    potassium = st.number_input("🪨 Potassium Level", min_value=0)
    soil = st.selectbox("🧑‍🌾 Soil Type", ['Acidic_Soil', 'Alkaline_Soil', 'Loamy_Soil', 'Neutral_Soil', 'Peaty_Soil'])

    if st.button("🧪 Recommend Fertilizer"):
        try:
            soil_cols = ['Acidic_Soil', 'Alkaline_Soil', 'Loamy_Soil', 'Neutral_Soil', 'Peaty_Soil']
            soil_vals = [1.0 if s == soil else 0.0 for s in soil_cols]
            crop_encoded = label_encoder_crop.transform([crop])[0]

            input_features = np.array([[temp, moist, rain, ph, nitrogen, phosphorous, potassium, carbon, *soil_vals, crop_encoded]])
            scaled_input = scaler_fert.transform(input_features)

            probs = rf_model.predict_proba(scaled_input)[0]

            top3_idx = np.argsort(probs)[::-1][:3]

            top3_ferts = label_encoder_fert.inverse_transform(top3_idx)

            st.markdown(f"""
            <div style='margin: 40px 0 30px 0; background-color:#1e1e1e;padding:20px 30px;border-radius:12px;
                        border-left:6px solid #4caf50;box-shadow:0 0 10px rgba(76, 175, 80, 0.4);'>
                <h3 style='color:#66ff66;'>🌱 Top 3 Recommended Fertilizers:</h3>
                <ul style='color:white;font-size:18px;line-height:1.6;'>
                    <li>🥇 <strong>{top3_ferts[0]}</strong></li>
                    <li>🥈 {top3_ferts[1]}</li>
                    <li>🥉 {top3_ferts[2]}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("📊 Fertilizer Input Overview")
            col_viz1, col_viz2 = st.columns(2)

            with col_viz1:
                st.markdown("#### 🌿 NPK Levels")
                npk_df = pd.DataFrame({
                    'Nutrient': ['Nitrogen', 'Phosphorous', 'Potassium'],
                    'Level': [nitrogen, phosphorous, potassium]
                })
                fig_npk = px.bar(npk_df, x='Nutrient', y='Level', color='Nutrient',
                                 color_discrete_sequence=px.colors.sequential.Viridis,
                                 title="NPK Composition", height=300)
                fig_npk.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_npk, use_container_width=True)

            with col_viz2:
                st.markdown("#### 🧪 Soil pH Indicator")
                fig_ph = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=ph,
                    title={'text': "Soil pH"},
                    gauge={
                        'axis': {'range': [0, 14]},
                        'bar': {'color': "Black"},
                        'steps': [
                            {'range': [0, 5.5], 'color': "#FF9999"},
                            {'range': [5.5, 7.5], 'color': "#A8FFB3"},
                            {'range': [7.5, 14], 'color': "#FFD580"},
                        ],
                    }
                ))
                fig_ph.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_ph, use_container_width=True)

            col_viz3, col_viz4 = st.columns(2)

            with col_viz3:
                st.markdown("#### 🌧️ Rainfall Overview")
                rain_color = "#00BFFF" if rain < 100 else "#FFD700" if rain < 300 else "#FF6347"
                fig_rain = go.Figure(go.Bar(
                    y=["Rainfall"],
                    x=[rain],
                    orientation='h',
                    marker_color=rain_color,
                    text=[f"{rain} mm"],
                    textposition='auto'
                ))
                fig_rain.update_layout(height=300, xaxis=dict(range=[0, 500]), title="Rainfall (mm)",
                                       margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_rain, use_container_width=True)

            with col_viz4:
                st.markdown("#### 🌿 Carbon Indicator")

                fig_carbon_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=carbon,
                    title={'text': "Carbon Level (0–5)", 'font': {'size': 18}},
                    gauge={
                        'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "darkgray"},
                        'bar': {'color': "black"},  # Needle/bar color
                        'bgcolor': "white",
                        'steps': [
                            {'range': [0, 1.5], 'color': "#FF6B6B"},    # Low
                            {'range': [1.5, 3.5], 'color': "#FFD93D"},  # Moderate
                            {'range': [3.5, 5], 'color': "#6BCB77"},    # Safe
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': carbon
                        }
                    }
                ))

                fig_carbon_gauge.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    font={'color': "White", 'family': "Arial"}
                )

                st.plotly_chart(fig_carbon_gauge, use_container_width=True)


            # --- Risk Indicator ---
            st.subheader("📉 Risk Indicator")
            if ph < 5 or ph > 8:
                st.warning("⚠️ Unfavorable Soil pH for Crop")
            elif rain < 50:
                st.warning("⚠️ Low Rainfall Risk")
            else:
                st.success("✅ Good Growing Conditions")

        except:
            st.warning("✨ Let's try that again! Some input values might be missing or out of range. Please review and submit once more — we're almost there! 😊")
        
