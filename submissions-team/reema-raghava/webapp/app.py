import os
import toml
from dotenv import load_dotenv
import streamlit as st
import requests
#from streamlit_extras.let_it_rain import rain

st.set_page_config(layout="wide")  # Optional: Set layout

# Define the path to config.toml in the subfolder
config_path = "./submissions-team/reema-raghava/webapp/.streamlit/config.toml"

# Check if the config file exists
if os.path.exists(config_path):
    # Load the config file manually
    with open(config_path, "r") as f:
        config = toml.load(f)

    # Extract theme settings if available
    if "theme" in config:
        theme = config["theme"]
        primary_color = theme.get("primaryColor", "#4b56ff")  # Default if not found
        background_color = theme.get("backgroundColor", "#ffffff")
        secondary_background_color = theme.get("secondaryBackgroundColor", "#f0f2f6")
        text_color = theme.get("textColor", "#000000")

        st.write("primary color from theme:", theme.get("primaryColor"))

        st.write("primary_color:", primary_color)

        # Apply theme dynamically using custom CSS
        st.markdown(
            f"""
            <style>
                :root {{
                    --primary-color: {primary_color};
                    --background-color: {background_color};
                    --secondary-background-color: {secondary_background_color};
                    --text-color: {text_color};
                }}
                body {{
                    background-color: var(--background-color) !important;
                    color: var(--text-color) !important;
                }}
                .stButton>button {{
                    background-color: var(--primary-color) !important;
                    color: white !important;
                }}
                .stSidebar {{
                    background-color: var(--secondary-background-color) !important;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.write("✅ Custom theme applied successfully!")
    else:
        st.write("⚠️ Theme section not found in config.toml")
else:
    st.write("⚠️ Config file not found at:", config_path)



load_dotenv()

# Get the BASE_URL from the environment variables
base_url = os.getenv("BASE_URL", "http://localhost:8000")

# Streamlit app title and description
st.title("Diamond Price Prediction")
st.write("Enter the values for each feature to predict if the tumor is benign or malignant.")

# Input fields
carat = st.number_input("Carat")
cut = st.radio("Cut", ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair'], horizontal=True)
color = st.radio("Color", ['D', 'E', 'F', 'G', 'H', 'I', 'J'], horizontal=True)
clarity = st.radio("Clarity", ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], horizontal=True)
x = st.number_input("Width (x)")
y = st.number_input("Length (y)")
z = st.number_input("Height (z)")
depth = st.number_input("Depth")
table = st.number_input("Table")

# Prediction button
if st.button("Predict"):
    # Prepare the data for API request
    input_data = {
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z,
    }

    # Make the API request
    response = requests.post(f"{base_url}/api/predict", json=input_data)

    if response.status_code == 200:
        result = response.json()
        # st.write(f"The prediction is {prediction['label']} (Class: {prediction['prediction']})")
        st.success(f"Predicted price: {result['prediction']}",
                   icon=":material/thumb_up:")
        #rain(emoji="🎈", font_size=54, falling_speed=5, animation_length="infinite", )
    else:
        print(response)
        st.write("Error: Could not retrieve prediction. Please try again.")

# Run the Streamlit App
# streamlit run app.py
