import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & SKY-GLASS THEME
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AuraSense AI", page_icon="🌬️", layout="wide")

st.markdown("""
    <style>
    /* Sky Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 50%, #7dd3fc 100%);
        color: #0f172a;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        color: #1e293b;
        margin-bottom: 20px;
    }

    /* Typography */
    .hero-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 4.5rem;
        background: linear-gradient(to right, #0369a1, #075985);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .tagline {
        font-size: 1.4rem;
        color: #0c4a6e;
        letter-spacing: 2px;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.5) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #0369a1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. ASSET LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_engine():
    try:
        model = pickle.load(open("aqi_model.pkl", "rb"))
        encoder = pickle.load(open("category_encoder.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        aqi_mapping = pickle.load(open("aqi_mapping.pkl", "rb"))
        feature_names = pickle.load(open("feature_names.pkl", "rb"))
        hist_avg = pickle.load(open("historical_avg.pkl", "rb"))
        return model, encoder, scaler, aqi_mapping, feature_names, hist_avg
    except Exception as e:
        st.error(f"Error initializing AuraSense: {e}")
        return [None]*6

model, encoder, scaler, aqi_mapping, feature_names, hist_avg = load_engine()

# -----------------------------------------------------------------------------
# 3. NAVIGATION & BRANDING
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color: #0369a1;'>🌬️ AuraSense</h2>", unsafe_allow_html=True)
    page = st.radio("Explore", ["🏠 Home", "🔍 Predictor"])
    st.markdown("---")
    st.caption("Precision forecasting for the air you breathe.")

# -----------------------------------------------------------------------------
# 4. PAGE: HOME
# -----------------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown('<div style="text-align: center; margin-top: 50px;">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">AuraSense</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Predicting the Air You Breathe.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card" style="height: 280px;">
            <h3 style="color: #0369a1;">🔬 Data-Driven Insight</h3>
            <p>AuraSense analyzes microscopic atmospheric pollutants using Random Forest Intelligence. By cross-referencing your location with seasonal chemical shifts, we provide high-fidelity air quality forecasts.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card" style="height: 280px;">
            <h3 style="color: #0369a1;">🛡️ Health First</h3>
            <p>Our goal is to help you navigate your environment safely. Whether it's morning jogs or weekend outings, AuraSense keeps you informed about the air that enters your lungs.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: #0369a1; margin-top: 30px;'>Atmospheric Indicators We Track</h3>", unsafe_allow_html=True)
    
    pulse_cols = st.columns(4)
    indicators = [
        {"icon": "🌫️", "name": "Particulates", "label": "PM2.5"},
        {"icon": "🚗", "name": "Emissions", "label": "NO₂ & CO"},
        {"icon": "☀️", "name": "Photochemical", "label": "O₃ (Ozone)"},
        {"icon": "🏭", "name": "Industrial", "label": "SO₂ Levels"}
    ]
    
    for i, item in enumerate(indicators):
        with pulse_cols[i]:
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.25); padding: 25px; border-radius: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.4);">
                    <span style="font-size: 2.5rem;">{item['icon']}</span>
                    <h4 style="color: #0369a1; margin: 10px 0 5px 0;">{item['name']}</h4>
                    <p style="font-size: 0.9rem; color: #0c4a6e; margin: 0;">{item['label']}</p>
                </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. PAGE: PREDICTOR
# -----------------------------------------------------------------------------
elif page == "🔍 Predictor":
    st.markdown("<h2 style='color: #0369a1;'>Atmospheric Forecast</h2>", unsafe_allow_html=True)
    
    # Input Container
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        s_state = st.selectbox("Current State", sorted(hist_avg["state"].unique()))
    with c2:
        s_season = st.selectbox("Current Season", sorted(hist_avg["season"].unique()))
    with c3:
        s_time = st.selectbox("Time of Day", sorted(hist_avg["time_of_day"].unique()))
    
    predict_btn = st.button("Generate Aura Report", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        filtered = hist_avg[(hist_avg["state"] == s_state) & 
                            (hist_avg["season"] == s_season) & 
                            (hist_avg["time_of_day"] == s_time)]

        if filtered.empty:
            st.error("No historical data available for this specific atmospheric profile.")
        else:
            with st.spinner("Decoding particles..."):
                hist_row = filtered.iloc[0]
                
                # Preprocessing
                input_data = {
                    "is_weekend": 0, "dew_point_c": hist_row["dew_point_c"], "heavy_rain": 0,
                    "cloud_cover_percent": hist_row["cloud_cover_percent"], "pm2_5_ugm3": hist_row["pm2_5_ugm3"],
                    "co_ugm3": hist_row["co_ugm3"], "no2_ugm3": hist_row["no2_ugm3"],
                    "so2_ugm3": hist_row["so2_ugm3"], "o3_ugm3": hist_row["o3_ugm3"],
                    "aod": hist_row["aod"], "festival_period": 0, "crop_burning_season": 0,
                    "state": s_state, "season": s_season, "time_of_day": s_time,
                }
                
                input_df = pd.DataFrame([input_data])
                categorical_cols = ["state", "season", "time_of_day"]
                encoded = encoder.transform(input_df[categorical_cols])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=input_df.index)
                final_input = pd.concat([input_df.drop(columns=categorical_cols), encoded_df], axis=1)[feature_names]
                
                continuous = ["pm2_5_ugm3", "co_ugm3", "no2_ugm3", "so2_ugm3", "o3_ugm3", "dew_point_c", "cloud_cover_percent", "aod"]
                final_input[continuous] = scaler.transform(final_input[continuous])

                prediction = model.predict(final_input)[0]
                prediction_proba = model.predict_proba(final_input)[0]
                result_cat = aqi_mapping[prediction]
                conf_score = max(prediction_proba) * 100

                # --- HEALTH ADVISORY LOGIC ---
                advisory_map = {
                    "Good": "🟢 **Safe to breathe.** Ideal for outdoor exercise and windows can remain open.",
                    "Moderate": "🟡 **Plan ahead.** Sensitive individuals should consider reducing heavy outdoor exertion.",
                    "Unhealthy for Sensitive Groups": "🟠 **Precaution advised.** Wear a mask if you have respiratory issues. Limit long outdoor stays.",
                    "Unhealthy": "🔴 **High Risk.** Wear an N95 mask outdoors and avoid strenuous physical activity.",
                    "Very Unhealthy": "🟣 **Hazardous.** Stay indoors, use air purifiers, and keep all windows tightly sealed."
                }
                tip = advisory_map.get(result_cat, "Monitor local news for air quality updates.")

                # --- RESULTS DISPLAY ---
                st.markdown("### Forecast Results")
                res_col1, res_col2 = st.columns([1.5, 1])
                
                with res_col1:
                    gauge_map = {"Good": 25, "Moderate": 75, "Unhealthy for Sensitive Groups": 125, "Unhealthy": 175, "Very Unhealthy": 250}
                    val = gauge_map.get(result_cat, 100)
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = val, 
                        title = {'text': f"Condition: {result_cat}", 'font': {'color': '#0369a1', 'size': 24}},
                        gauge = {
                            'axis': {'range': [0, 300], 'tickcolor': "#0369a1"},
                            'bar': {'color': "#0ea5e9"},
                            'steps': [
                                {'range': [0, 50], 'color': "#bbf7d0"},
                                {'range': [50, 100], 'color': "#fef08a"},
                                {'range': [100, 300], 'color': "#fecaca"}]}))
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#0369a1"}, height=350)
                    st.plotly_chart(fig, use_container_width=True)

                with res_col2:
                    st.metric("Model Confidence", f"{conf_score:.1f}%")
                    st.markdown(f"""
                        <div style="background: rgba(255, 255, 255, 0.6); padding: 25px; border-radius: 20px; border-left: 5px solid #0ea5e9; margin-top: 20px;">
                            <h4 style="margin:0; color: #0369a1;">🛡️ AI Health Advisory</h4>
                            <p style="margin-top:10px; font-size: 1.1rem;">The air quality is <b>{result_cat}</b>.</p>
                            <div style="background: rgba(14, 165, 233, 0.1); padding: 15px; border-radius: 12px; color: #0c4a6e; border: 1px dashed #0ea5e9;">
                                {tip}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                # Pollutant Breakdown
                st.markdown("---")
                st.subheader("Predicted Pollutant Profile")
                p_df = pd.DataFrame({
                    "Pollutant": ["PM2.5", "CO", "NO₂", "SO₂", "O₃"],
                    "Level": [hist_row["pm2_5_ugm3"], hist_row["co_ugm3"], hist_row["no2_ugm3"], hist_row["so2_ugm3"], hist_row["o3_ugm3"]]
                })
                fig_bar = px.bar(p_df, x="Pollutant", y="Level", color="Level", color_continuous_scale="Blues")
                fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font={'color': "#0369a1"})
                st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("<br><p style='text-align: center; color: #64748b; font-size: 0.8rem;'>AuraSense AI | Predicted Intelligence © 2026</p>", unsafe_allow_html=True)