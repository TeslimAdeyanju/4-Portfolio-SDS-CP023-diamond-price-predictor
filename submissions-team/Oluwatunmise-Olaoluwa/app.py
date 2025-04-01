import streamlit as st
import joblib
import numpy as np
import pandas as pd


# Define the class BEFORE loading the pickle file
#Define the Ensemble Model Class

class EnsembleModel:
    def __init__(self, rf_model, xgb_model):
        self.rf_model = rf_model
        self.xgb_model = xgb_model

    def fit(self, X, y):
        """Train both models on the given data."""
        self.rf_model.fit(X, y)
        self.xgb_model.fit(X, y)    

    def predict(self, X):
        rf_preds = self.rf_model.predict(X)
        xgb_preds = self.xgb_model.predict(X)
        ensemble_preds = (rf_preds + xgb_preds) / 2  # Averaging predictions
        return ensemble_preds


# Load the trained model using Joblib
model_filename = "submissions-team/Oluwatunmise-Olaoluwa/diamond_price_model.joblib"
model = joblib.load(model_filename)


# Define the app title
st.title("💎 Diamond Price Predictor")

# Sidebar for user input
st.sidebar.header("Enter Diamond Attributes")

# Define color labels with descriptions
color_options = {
    "J (Most yellow, least valuable)": "J",
    "I (Slight yellow tint)": "I",
    "H (Faint color)": "H",
    "G (Noticeable tint begins)": "G",
    "F (Near colorless)": "F",
    "E (Very slightly colorless)": "E",
    "D (Completely colorless, most valuable)": "D",
}

# Define clarity labels with descriptions
clarity_options = {
    "I1 (Visible Inclusions)": "I1",
    "SI2 (Minor Inclusions)": "SI2",
    "SI1 (Small Inclusions)": "SI1",
    "VS2 (Slight Inclusions)": "VS2",
    "VS1 (Very Small Inclusions)": "VS1",
    "VVS2 (Tiny Inclusions)": "VVS2",
    "VVS1 (Nearly Invisible Inclusions)": "VVS1",
    "IF (Flawless - No Inclusions)": "IF",
}

# Collect user inputs
carat = st.sidebar.slider("Carat", 0.2, 2.0, 0.5)
cut_quality = st.sidebar.selectbox("Cut Quality", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.sidebar.selectbox("Color (Diamond Tint Level)", list(color_options.keys()))
clarity = st.sidebar.selectbox("Clarity (Diamond Purity Level)", list(clarity_options.keys()))
depth = st.sidebar.slider("Depth (%) (Total Height of Diamond)", 50.0, 75.0, 61.5)
table = st.sidebar.slider("Table (%) (Top Surface Width of Diamond)", 50.0, 75.0, 57.0)
x = st.sidebar.slider("X (Length mm)", 3.0, 10.0, 5.5)
y = st.sidebar.slider("Y (Width mm)", 3.0, 10.0, 5.5)
z = st.sidebar.slider("Z (Depth mm - Height from Top to Bottom)", 2.0, 6.0, 3.5)

# Convert categorical features to numeric (you need to use the same encoding as in training)
cut_mapping = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
color_mapping = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
clarity_mapping = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}

cut = cut_mapping[cut_quality]
color = color_mapping[color_options[color]]  # Convert user selection to corresponding label
clarity = clarity_mapping[clarity_options[clarity]]

# 🔹 Feature Engineering (Apply the same transformations as in training)
carat_color_interaction = carat * color
carat_clarity_interaction = carat * clarity
quality_composite = cut*0.2 + color*0.4 + clarity*0.4
carat_quality_interaction = carat * quality_composite

# Prepare the transformed input features (matching model training)
input_features = np.array([[carat, carat_color_interaction, carat_clarity_interaction, quality_composite, carat_quality_interaction]])

# Predict the price
if st.sidebar.button("Predict Price 💰"):
    predicted_price = model.predict(input_features)[0]
    st.success(f"💎 Estimated Diamond Price: **${predicted_price:,.2f}**")
