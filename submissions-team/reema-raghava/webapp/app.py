import os
from dotenv import load_dotenv
import streamlit as st
import requests
#from streamlit_extras.let_it_rain import rain

# load environment variables from .env file
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