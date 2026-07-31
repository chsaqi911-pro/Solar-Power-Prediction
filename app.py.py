import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Solar Power Generation Prediction",
    page_icon="☀️",
    layout="wide"
)

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------
try:
    model = joblib.load("solar_power_prediction_model.pkl")
    features = joblib.load("feature_names.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ----------------------------------------------------
# Title
# ----------------------------------------------------
st.title("☀️ Solar Power Generation Prediction System")

st.markdown("""
Predict **hourly solar power generation** using a **Random Forest Regression** model.

The model was trained using historical weather observations and solar power generation data.
""")

st.markdown("---")

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
st.sidebar.header("⚙️ Weather Parameters")

# Weather Inputs
windspeed = st.sidebar.slider(
    "Wind Speed (m/s)",
    min_value=0.0,
    max_value=30.0,
    value=3.0,
    step=0.1
)

sunshine = st.sidebar.slider(
    "Sunshine (hours)",
    min_value=0.0,
    max_value=15.0,
    value=6.0,
    step=0.1
)

airpressure = st.sidebar.slider(
    "Air Pressure (hPa)",
    min_value=950.0,
    max_value=1050.0,
    value=1013.0,
    step=0.1
)

radiation = st.sidebar.slider(
    "Solar Radiation (W/m²)",
    min_value=0.0,
    max_value=900.0,
    value=300.0,
    step=1.0
)

airtemperature = st.sidebar.slider(
    "Air Temperature (°C)",
    min_value=-10.0,
    max_value=50.0,
    value=25.0,
    step=0.1
)

humidity = st.sidebar.slider(
    "Relative Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=0.1
)

st.sidebar.markdown("---")

# Date and Time
selected_date = st.sidebar.date_input("Select Date")

hour = st.sidebar.selectbox(
    "Hour of Day",
    list(range(24))
)

month = selected_date.month
day = selected_date.day

# ----------------------------------------------------
# Two Column Layout
# ----------------------------------------------------
left, right = st.columns([2,1])

with left:

    st.subheader("📋 Input Summary")

    input_df = pd.DataFrame({
        "Parameter":[
            "Wind Speed",
            "Sunshine",
            "Air Pressure",
            "Solar Radiation",
            "Air Temperature",
            "Relative Humidity",
            "Month",
            "Day",
            "Hour"
        ],
        "Value":[
            windspeed,
            sunshine,
            airpressure,
            radiation,
            airtemperature,
            humidity,
            month,
            day,
            hour
        ]
    })

    st.dataframe(input_df, use_container_width=True)

with right:

    st.subheader("ℹ️ Model Information")

    st.info("""
**Algorithm**

Random Forest Regressor

**Input Features**

9

**Target**

Hourly Solar Power Generation

**Best R² Score**

0.811

**Training Samples**

3278
""")

st.markdown("---")

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------
if st.button("🔮 Predict Solar Power Generation"):

    # Hybrid Engineering Rule
    if radiation <= 0 or sunshine <= 0:

        prediction = 0.0

    else:

        input_data = np.array([[
            windspeed,
            sunshine,
            airpressure,
            radiation,
            airtemperature,
            humidity,
            month,
            day,
            hour
        ]])

        prediction = model.predict(input_data)[0]

        # Prevent negative prediction
        prediction = max(prediction, 0)

    st.success("Prediction Completed Successfully")

    st.metric(
    	label="Predicted Solar Power Generation",
    	value=f"{prediction:.2f} kWh"

    )

    st.caption("Hourly predicted system production")

st.markdown("---")

# ----------------------------------------------------
# About
# ----------------------------------------------------
with st.expander("📖 About this Application"):

    st.write("""
This application predicts **hourly solar power generation**
using a **Random Forest Machine Learning model**.

### Workflow

1. User enters weather parameters.
2. Date is converted to Month and Day.
3. If Radiation ≤ 0 or Sunshine ≤ 0,
   the application directly predicts **0**.
4. Otherwise, the trained Random Forest model predicts
   hourly solar power generation.

### Input Features

- Wind Speed
- Sunshine
- Air Pressure
- Solar Radiation
- Air Temperature
- Relative Humidity
- Month
- Day
- Hour

### Output

Predicted hourly solar power generation.
""")

st.markdown("---")

st.markdown(
"""
<center>

Developed using **Python • Scikit-Learn • Streamlit • Random Forest Regression**

</center>
""",
unsafe_allow_html=True
)