import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("C:/Users/adann/Desktop/adagithub_clone/SDS-CP023-diamond-price-predictor/submissions-team/adanna-alutu/final_xgboost_model.pkl")
encoder = joblib.load("C:/Users/adann/Desktop/adagithub_clone/SDS-CP023-diamond-price-predictor/submissions-team/adanna-alutu/Ordinal_encoder.pkl")
scaler = joblib.load("C:/Users/adann/Desktop/adagithub_clone/SDS-CP023-diamond-price-predictor/submissions-team/adanna-alutu/Standard_scaler.pkl")



st.title("Ada's Diamond project") 
col9, col10 = st.columns(2)
with col9:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRHayOdTkOZWzXQm2vUqxtw0h_bCMw6bKYCZwrfLSF3YF1UREJ5Aogqbw&s")
with col10:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYY5YtG5uhHfbpqaEULiDK3kOXu8ky4bOlAjsZiU6ur-CWYe7zrmprpw&s")
    

#sl_carat = st.slider('Carat - choose a range', 0.2, 5.01)
#to put in equal sized columns

col1, col2, col3 = st.columns(3)

with col1:
    carat = st.slider('Carat - choose a range', 0.2, 5.01)

with col2:
    cut = st.selectbox('CUT/shine selections', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])

with col3:
    color = st.selectbox('Color appeal', ['D', 'E', 'F', 'G', 'H', 'I', 'J'])

col4, col5 = st.columns(2)

with col4:
    depth = st.slider('Depth - choose a range', 43, 79)

with col5:
    clarity = st.selectbox('Clarity selections', ['SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'Other'])

col6, col7 = st.columns(2)
with col6:
    table = st.slider('Table-diamond size', 43, 95)
with col7:
    volume = st.slider('Volume-length,height, width', 0, 3841)

btn = st.button('Predict best Price')
if btn:
    st.write('standby')
    in_data = pd.DataFrame({
        'volumedim_xyz': [volume],
        'cut': [cut],
        'color': [color],
        'clarity_other': [clarity],
        'carat': [carat],
        'depth': [depth],
        'table': [table]
    })
    indata_enc = encoder.transform(in_data[['cut', 'color', 'clarity_other']])
    indata_enc_df = pd.DataFrame(indata_enc, columns=['cut', 'color', 'clarity_other'])
    indata_allcolumns = pd.concat([in_data[['volumedim_xyz']],indata_enc_df, in_data[['carat', 'depth', 'table']].reset_index(drop = True)], axis = 1)
    indata_scaled = scaler.transform(indata_allcolumns)
    prediction = model.predict(indata_scaled)
    #st.write(prediction)
   # st.success = (f"Best value: ${prediction[0]:,.2f}")
    st.write(f"Best value: ${prediction[0]:,.2f}")

#sidebar
st.sidebar.title('SDS-CP023-Diamond Price Prediction')
st.sidebar.header('Problem Definition')
st.sidebar.subheader('The Context')
st.sidebar.text('Why is this problem important to solve? To build a model that will predict the price of diamond based on the following features/attributes - carat, cut, color, clarity, depth, and dimensions.')
st.sidebar.subheader('The objective:')
st.sidebar.text('What is the intended goal? To predict price that matches the actual price 95% of the time.')
st.sidebar.subheader('Data Description:')
#st.sidebar.button('Click nthis')

st.sidebar.text('Price: cost of diamond in US dollars $')
st.sidebar.text('carat: weight of the diamond')
st.sidebar.text("cut: represents the 'shine' with the following categories: ideal, premium, good, very good and fair")
st.sidebar.text('color: the color is for visual appeal. Values range from D to J')
st.sidebar.text('clarity: explains the internal diamond purity with these classifications - SI2 SI1 VS1 VS2 VVS2 VVS1 and I1')
st.sidebar.text('depth: represents the diamond depth percentage')
st.sidebar.text('table: measure of flatness at the top')
st.sidebar.text('dimensions: different measurements from the flat surface - x = width, y = length, z = height')



#import time
#with st.spinner('wait for 3 seconds and then do whatever'):
#    time.sleep(3)
#st.write("thank you")

#st.balloons()
