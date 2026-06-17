import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="EV Range Predictor", page_icon="⚡", layout="wide")

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.markdown("""
<style>
    .main { background-color: #1a1a2e; }
    .stApp { background-color: #1a1a2e; color: white; }
    h1, h2, h3, p, label { color: white !important; }
    .metric-card {
        background-color: #0f0f1a;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00D2BE;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ EV Battery Range Predictor")
st.markdown("**Predict how far your electric vehicle will go based on battery and driving conditions**")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Battery Specs")
    battery_capacity = st.slider("Battery Capacity (kWh)", 20.0, 120.0, 75.0, 0.5)
    battery_health = st.slider("Battery Health (%)", 70.0, 100.0, 90.0, 0.5)
    charge_cycles = st.slider("Charge Cycles", 0, 2000, 500, 10)
    charging_power = st.slider("Charging Power (kW)", 50.0, 200.0, 100.0, 5.0)

with col2:
    st.subheader("Driving Conditions")
    avg_speed = st.slider("Average Speed (km/h)", 20.0, 150.0, 80.0, 1.0)
    energy_consumption = st.slider("Energy Consumption (kWh/100km)", 10.0, 30.0, 18.0, 0.5)
    temperature = st.slider("Temperature (°C)", -20.0, 45.0, 20.0, 1.0)
    mileage = st.slider("Total Mileage (km)", 5000, 250000, 50000, 1000)

# Feature engineering - same as training
efficiency_score = battery_capacity / energy_consumption
degradation_factor = (100 - battery_health) / 100
power_capacity_ratio = charging_power / battery_capacity
temp_impact = abs(temperature - 20)
usage_intensity = mileage / (charge_cycles + 1)
speed_efficiency = avg_speed / energy_consumption

features = np.array([[
    battery_capacity, battery_health, temperature, avg_speed,
    energy_consumption, charge_cycles, mileage, charging_power,
    efficiency_score, degradation_factor, power_capacity_ratio,
    temp_impact, usage_intensity, speed_efficiency
]])

prediction = model.predict(features)[0]
prediction = max(50, min(750, prediction))

st.markdown("---")
st.subheader("Predicted Range")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Range", f"{prediction:.0f} km")
with col2:
    effective = prediction * (battery_health / 100)
    st.metric("Effective Range", f"{effective:.0f} km", 
              delta=f"{effective - prediction:.0f} km (health impact)")
with col3:
    highway_range = prediction * 0.85
    st.metric("Highway Estimate", f"{highway_range:.0f} km")

st.markdown("---")

# Gauge chart
fig, ax = plt.subplots(figsize=(8, 3))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

max_range = 750
pct = prediction / max_range

color = '#00D2BE' if pct > 0.5 else '#FFD700' if pct > 0.3 else '#E8002D'
ax.barh(['Range'], [prediction], color=color, height=0.4)
ax.barh(['Range'], [max_range], color='#2a2a4e', height=0.4)
ax.barh(['Range'], [prediction], color=color, height=0.4)
ax.set_xlim(0, max_range)
ax.set_xlabel('Range (km)', color='white')
ax.tick_params(colors='white')
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.set_title(f'Predicted Range: {prediction:.0f} km out of {max_range} km max', 
             color='white', fontsize=12)
st.pyplot(fig)

st.markdown("---")
st.subheader("Key Insights")

col1, col2 = st.columns(2)
with col1:
    if temperature < 0 or temperature > 35:
        st.warning(f"Extreme temperature ({temperature}°C) is reducing your range. Optimal range is 15-25°C.")
    if battery_health < 80:
        st.warning(f"Battery health at {battery_health}% - consider battery service to restore range.")
    if avg_speed > 120:
        st.warning("High speed significantly increases energy consumption and reduces range.")

with col2:
    st.info(f"Efficiency Score: {efficiency_score:.2f} kWh/100km ratio")
    st.info(f"Battery Degradation: {degradation_factor*100:.1f}% capacity lost")
    if charge_cycles > 1500:
        st.warning(f"High charge cycles ({charge_cycles}) may be affecting battery performance.")

st.markdown("---")
st.caption("Built with Python, XGBoost, and Streamlit | Data: Electric Vehicle Analytics Dataset")