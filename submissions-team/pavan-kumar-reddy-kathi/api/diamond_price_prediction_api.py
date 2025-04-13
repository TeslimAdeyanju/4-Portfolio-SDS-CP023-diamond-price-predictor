import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import os
from enums import Cut,Color,Clarity


# Instantiate FastAPI
app = FastAPI(title='Diamond Price Predictor',
              description='API to predict diamond price based on its features',
              version='1.0.0')

# Load Prediction model
model = joblib.load(os.path.join(os.path.dirname(__file__),'model','prediction_model.pkl'))

# Build encoding for Cut, Color, Clarity

# order all below categories from worst to best
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

cut_mapping = {cut: idx for idx, cut in enumerate(cut_order)}
color_mapping = {color: idx for idx, color in enumerate(color_order)}
clarity_mapping = {clarity: idx for idx, clarity in enumerate(clarity_order)}

# Input request/payload structure
class DiamondFeatures(BaseModel):
    cut:Cut
    color:Color
    clarity:Clarity
    carat:float
    depth:float
    table:float
    width:float
    height:float
    length:float

@app.post(
    path='/api/predict-diamond-price',
    summary='Predicts diamond price',
    description='Predicts diamond price based on its features',
    tags=["Prediction"]
)
def predict_diamond_price(request:DiamondFeatures):
    matrix_feature = np.array([[
        request.carat,
        request.depth,
        request.table,
        request.width,
        request.height,
        request.length,
        cut_mapping[request.cut],
        color_mapping[request.color],
        clarity_mapping[request.clarity]
    ]])

    y_predict = model.predict(matrix_feature)
    return y_predict[0]
