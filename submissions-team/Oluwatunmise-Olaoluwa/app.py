import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image


# Define the dataset path
dataset_path = "/Users/olaoluwatunmise/Diamond-Price-Predictor/SDS-CP023-diamond-price-predictor/submissions-team/Oluwatunmise-Olaoluwa/type-of-the-Diamond.csv"

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv(dataset_path)

df = load_data()


#df = load_data()

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


# Load the trained model
model_filename = "submissions-team/Oluwatunmise-Olaoluwa/diamond_price_model.joblib"
model = joblib.load(model_filename)

# Set up the app title
st.title("💎 Diamond Price Predictor")

# Load and display the image
image_path =  "/Users/olaoluwatunmise/Diamond-Price-Predictor/SDS-CP023-diamond-price-predictor/submissions-team/Oluwatunmise-Olaoluwa/Diamond.jpg" #"/mnt/data/viktor-mindt-4yWWIEKOBaU-unsplash.jpg"
image = Image.open(image_path)
st.image(image, use_container_width=True)  # ✅ Fixed deprecated parameter

#Theme Color
theme = st.sidebar.radio("🌗 Choose Theme", ["Light", "Dark"])


st.subheader("🔎 Model Information")
st.info(
    "This model uses a combination of **Random Forest** and **XGBoost** "
    "to predict the price of a diamond based on its attributes."
)

st.subheader("📖 How to Use")
st.markdown(
    """
    1️⃣ Adjust the **Carat**, **Color**, **Clarity**, and **Cut** options.  
    2️⃣ Click **Predict Price 💰** to get an estimate.  
    3️⃣ Check the **Dataset Insights** for more details.  
    """
)

# Compute realistic insights
avg_price = df["price"].mean()  # Average diamond price
most_common_carat = df["carat"].mode()[0]  # Most frequent cut category
max_price = df["price"].max()  # Maximum price recorded

# Display dataset insights
if st.sidebar.checkbox("Show Dataset Insights"):
    st.sidebar.write(f"💰 **Average Diamond Price:** ${avg_price:,.2f}")
    st.sidebar.write(f"🔷 **Most Common Carat:** {most_common_carat}")
    st.sidebar.write(f"🔺 **Max Price Recorded:** ${max_price:,.2f}")

# About the diamond
st.sidebar.markdown("## 💎 About the Diamond Price Predictor")
st.sidebar.info(
    "Welcome to the **Diamond Price Predictor**, an advanced machine learning-powered application designed to estimate the price of diamonds based on their unique attributes. This app leverages cutting-edge regression models trained on real-world diamond data to provide accurate price predictions, helping buyers, sellers, and investors make informed decisions. \n\n"
   
)

st.sidebar.markdown("## 🔍 How It Works")
st.sidebar.info(
    " Our model is built on the 4C framework used in the diamond industry: \n\n"
     "🔹 **Carat** - Diamond weight.\n"
    "🔹 **Cut** - The quality of the cut, affecting light reflection.\n"
    "🔹 **Color** - The presence of a yellowish tint.\n"
    "🔹 **Clarity** - Internal and external imperfections.\n\n"

    " Additionally, we analyze depth, table percentage, and dimensional factors (X, Y, Z measurements) to further refine our predictions."
    )

st.sidebar.markdown("## 📊 Powered by Machine Learning")
st.sidebar.info(
    " **The app utilizes multiple machine learning models, including Random Forest, XGBoost, and Gradient Boosting, to ensure the most accurate pricing estimates. After extensive feature engineering, hyperparameter tuning, and model validation, the best-performing model has been integrated into this application.** \n\n"
)


st.sidebar.markdown("## 🌟 Features of the App")
st.sidebar.info(
    
    "✅ **Real-time Price Estimation - Instantly predicts diamond prices based on user inputs.**  \n"
    "✅ **User-Friendly Interface - Simple and intuitive design for easy navigation.**  \n"
    "✅ **Insights & Visualizations - Gain a deeper understanding of diamond attributes and their impact on pricing.**  \n"
    "✅ **Deployed on Streamlit - Accessible from anywhere with a seamless experience.** \n\n"
)

st.sidebar.markdown("## 🚀 Why Use This App?")
st.sidebar.info( 
   " ✔ **For Buyers - Check if a diamond is fairly priced before purchasing.**  \n"
   " ✔ **For Sellers - Get price recommendations to optimize sales.**  \n"
   " ✔ **For Investors - Assess market trends and diamond valuations.**  \n\n"
)

st.sidebar.markdown("## 🛠️ Technical Details")
st.sidebar.info( 
    "📊 **Github Repo:** [Diamond Prediction Model](https://github.com/apostleoffinance/SDS-CP023-diamond-price-predictor/tree/main/submissions-team/Oluwatunmise-Olaoluwa) \n"
    "🛠 **Tech Stack:** Pandas, NumPy, Scikit-learn, XGBoost, Seaborn, Matplotlib, Streamlit \n"
    "📏 **Evaluation Metrics:** RMSE, MAE, R² \n\n"
    "Start exploring diamond prices with **data-driven confidence**! 💎✨"
)


# Define input options
color_options = {
    "J (Most yellow, least valuable)": "J",
    "I (Slight yellow tint)": "I",
    "H (Faint color)": "H",
    "G (Noticeable tint begins)": "G",
    "F (Near colorless)": "F",
    "E (Very slightly colorless)": "E",
    "D (Completely colorless, most valuable)": "D",
}

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

# Collect user inputs (arranged in 3-column layout for better alignment)
col1, col2, col3 = st.columns(3)

with col1:
    carat = st.slider("Carat", 0.2, 2.0, 0.5)
    color = st.selectbox("Color (Diamond Tint Level)", list(color_options.keys()))
    x = st.slider("X (Length mm)", 3.0, 10.0, 5.5)

with col2:
    cut_quality = st.selectbox("Cut Quality", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    clarity = st.selectbox("Clarity (Diamond Purity Level)", list(clarity_options.keys()))
    y = st.slider("Y (Width mm)", 3.0, 10.0, 5.5)

with col3:
    depth = st.slider("Depth (%)", 50.0, 75.0, 61.5)
    table = st.slider("Table (%)", 50.0, 75.0, 57.0)
    z = st.slider("Z (Depth mm)", 2.0, 6.0, 3.5)

# Convert categorical features to numeric
cut_mapping = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
color_mapping = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
clarity_mapping = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}

cut = cut_mapping[cut_quality]
color = color_mapping[color_options[color]]
clarity = clarity_mapping[clarity_options[clarity]]

# Feature Engineering
carat_color_interaction = carat * color
carat_clarity_interaction = carat * clarity
quality_composite = cut * 0.2 + color * 0.4 + clarity * 0.4
carat_quality_interaction = carat * quality_composite

# Prepare input features
input_features = np.array([[carat, carat_color_interaction, carat_clarity_interaction, quality_composite, carat_quality_interaction]])


# Predict the price
if st.button("Predict Price 💰"):
    predicted_price = model.predict(input_features)[0]
    st.success(f"💎 Estimated Diamond Price: **${predicted_price:,.2f}**")

# Compare Two Diamond
if st.button("Compare Two Diamonds"):
    st.warning("Comparison feature coming soon! 🚀")