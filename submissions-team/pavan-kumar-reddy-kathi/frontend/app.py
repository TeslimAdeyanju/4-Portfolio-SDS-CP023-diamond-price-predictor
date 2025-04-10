import os
from dotenv import load_dotenv
import streamlit as st
import requests
from streamlit_extras.let_it_rain import rain
import logging

# load env variables from .env file
load_dotenv()

# print(prediction_url)
st.title('Diamond Price Predictor')
st.header('Enter values for each feature to predict diamond price')

# order all below categories from worst to best
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

cut_mapping = {cut: idx for idx, cut in enumerate(cut_order)}
color_mapping = {color: idx for idx, color in enumerate(color_order)}
clarity_mapping = {clarity: idx for idx, clarity in enumerate(clarity_order)}

# drop down options for cut/color/clarity
selected_cut_key = st.selectbox('Cut', cut_mapping.keys())
selected_cut_value = cut_mapping[selected_cut_key]

selected_color_key = st.selectbox('Color', color_mapping.keys())
selected_color_value = color_mapping[selected_color_key]

selected_clarity_key = st.selectbox('Clarity', clarity_mapping.keys())
selected_clarity_value = clarity_mapping[selected_clarity_key]

# text inputs
carat = st.number_input('Carat')

depth = st.number_input('Depth')

table = st.number_input('Table')

width = st.number_input('Width')

height = st.number_input('Height')

length = st.number_input('Length')

if st.button("Predict"):
    input_data = {
        'cut': selected_cut_value,
        'color': selected_color_value,
        'clarity': selected_clarity_value,
        'carat': carat,
        'depth': depth,
        'table': table,
        'width': width,
        'height': height,
        'length': length
    }

    prediction_url = os.getenv('PREDICTION_API_URL')
    response = requests.post(f'{prediction_url}/api/predict-diamond-price', json = input_data)

    if response.status_code == 200:
        predicted_price = response.text
        st.success(f'**Predicted Price:${predicted_price}**', icon='✔')
        rain(emoji='💰', font_size=50, falling_speed=5, animation_length='infinite')
    else:
        logging.error(f'Error response when calling api-{response.json()} '
                      f'with request{input_data}')
        st.error('Invalid HTTP Status Code. Please retry with valid values')

