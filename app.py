import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("models/housing_pipeline.joblib")

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Error loading model. Make sure you ran 'python src/train.py' first. Error: {e}")
    st.stop()

st.title("🏠 California Housing Price Prediction App")
st.write("Enter the neighborhood metrics below to receive an estimated house value.")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    med_inc = st.number_input("Median Income ($10,000s)", min_value=0.5, max_value=15.0, value=3.87, step=0.1)
    house_age = st.slider("House Age (Years)", min_value=1, max_value=52, value=28)
    ave_rooms = st.number_input("Average Rooms per Household", min_value=1.0, max_value=20.0, value=5.4, step=0.1)

with col2:
    ave_bedrms = st.number_input("Average Bedrooms per Household", min_value=0.5, max_value=10.0, value=1.1, step=0.1)
    population = st.number_input("Block Population", min_value=3, max_value=35000, value=1425, step=10)
    ave_occup = st.number_input("Average Household Occupancy", min_value=1.0, max_value=10.0, value=3.0, step=0.1)

with col3:
    latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=35.6, step=0.01)
    longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-119.5, step=0.01)

st.markdown("---")

if st.button("🔮 Predict House Price", type="primary"):
    input_data = pd.DataFrame([[
        med_inc, house_age, ave_rooms, ave_bedrms, 
        population, ave_occup, latitude, longitude
    ]], columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])

    prediction = pipeline.predict(input_data)[0]
    estimated_price = prediction * 100000

    st.success(f"### Estimated Price: **${estimated_price:,.2f}**")