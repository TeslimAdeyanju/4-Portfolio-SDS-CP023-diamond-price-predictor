import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("/Users/sot/SDS-CP023-diamond-price-predictor/submissions-team/Patrick-Edosoma/final_gradient_boosting_model.pkl")
encoder = joblib.load("/Users/sot/SDS-CP023-diamond-price-predictor/submissions-team/Patrick-Edosoma/Ordinal_encoder.pkl")
scaler = joblib.load("/Users/sot/SDS-CP023-diamond-price-predictor/submissions-team/Patrick-Edosoma/Standard_scaler.pkl")

st.sidebar.title("About the App")
st.sidebar.info(
    "This app predicts the price of diamonds based on their characteristics such as cut, color, clarity, carat, depth, table, and volume. "
    "It uses a trained Gradient Boosting model with preprocessing steps that include ordinal encoding and feature scaling. "
    "The model achieved a cross-validation score of **96%**, indicating strong predictive performance and reliability."
)

st.sidebar.title("Features")
st.sidebar.write("""
- **Dataset Overview:**  
  The dataset contains **53,940 records**. Each record represents a diamond with multiple attributes describing its physical characteristics and price in USD.
                 
  - **Carat:** Weight of the diamond.
  - **Cut:** Determines shine (Ideal, Premium, Very Good, Good, Fair).
  - **Color:** Graded from D (best) to J (lowest), affecting visual appeal.
  - **Clarity:** Internal purity (VVS1, VVS2, VS1, VS2, SI1, SI2, Others).
  - **Depth:** Diamond’s depth percentage, affecting light reflection.
  - **Table:** Flat top surface area, influences light entry and brilliance.
  - **X, Y, Z:** Physical dimensions – width, length, height.
  - **Volume:** Engineered feature from X, Y, Z to address multicollinearity.

These features work together to determine a diamond’s value. A larger diamond with poor cut may be worth less than a smaller, well-cut diamond.
""")
st.markdown(" ## 💎 Welcome to Diamond Price Predictor 💎")


st.image("/Users/sot/SDS-CP023-diamond-price-predictor/submissions-team/Patrick-Edosoma/diamond copy.png", caption="Diamond Prediction App", use_container_width=True)

st.header("Enter Diamond Characteristics")

col1, col2, col3 = st.columns(3)

with col1:
    cut = st.selectbox("Cut", options=["Fair", "Good", "Very Good", "Premium", "Ideal"])
    carat = st.number_input("Carat", min_value=0.0, value=0.5, step=0.01)

with col2:
    color = st.selectbox("Color", options=["D", "E", "F", "G", "H", "I", "J"])
    depth = st.number_input("Depth", min_value=0.0, value=60.0, step=0.1)

with col3:
    clarity = st.selectbox("Clarity", options=["VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "Others"])
    table = st.number_input("Table", min_value=0.0, value=60.0, step=0.1)

volume = st.number_input("Volume", min_value=0.0, value=50.0, step=0.1)

if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'cut': [cut],
        'color': [color],
        'clarity': [clarity],
        'carat': [carat],
        'depth': [depth],
        'table': [table],
        'volume': [volume]
    })
    input_encoded = encoder.transform(input_data[['cut', 'color', 'clarity']])
    input_encoded_df = pd.DataFrame(input_encoded, columns=['cut', 'color', 'clarity'])
    input_combined = pd.concat([input_encoded_df, input_data[['carat', 'depth', 'table', 'volume']].reset_index(drop=True)], axis=1)
    input_scaled = scaler.transform(input_combined)
    prediction = model.predict(input_scaled)
    st.success(f"Estimated Diamond Price: ${prediction[0]:,.2f}")
