import os
from dotenv import load_dotenv
import streamlit as st
import requests
from streamlit_extras.let_it_rain import rain
import logging

# load env variables
load_dotenv()

st.title('Diamond Price Predictor')
st.header('Enter values for each feature to predict diamond price')

# order all below categories from worst to best
cut_options = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_options = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_options = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

# drop down options for cut/color/clarity
selected_cut = st.selectbox('Cut', cut_options)

selected_color = st.selectbox('Color', color_options)

selected_clarity = st.selectbox('Clarity', clarity_options)

# text inputs
carat = st.number_input('Carat')

depth = st.number_input('Depth')

table = st.number_input('Table')

width = st.number_input('Width')

height = st.number_input('Height')

length = st.number_input('Length')

if st.button("Predict"):
    input_data = {
        'cut': selected_cut,
        'color': selected_color,
        'clarity': selected_clarity,
        'carat': carat,
        'depth': depth,
        'table': table,
        'width': width,
        'height': height,
        'length': length
    }

    if carat <= 0 or depth <= 0 or table <= 0 \
        or width <= 0 or height <= 0 or length <=0 :
        st.error('Please enter values greater than zero.')
    else:
        prediction_url = os.getenv('PREDICTION_API_URL')
        response = requests.post(f'{prediction_url}/api/predict-diamond-price', json = input_data)

        if response.status_code == 200:
            predicted_price = response.text
            st.success(f'**Predicted Price:${round(float(predicted_price),2)}**', icon='✔')
            rain(emoji='💰', font_size=50, falling_speed=5, animation_length='infinite')
        else:
            logging.error(f'Error response when calling api-{response.json()} '
                          f'with request{input_data}')
            st.error('Invalid HTTP Status Code. Please retry with valid values')

