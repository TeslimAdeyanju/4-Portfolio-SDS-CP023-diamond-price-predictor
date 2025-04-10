import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import os


# Instantiate FastAPI
app = FastAPI(title='Diamond Price Predictor',
              description='API to predict diamond price based on its features',
              version='1.0.0')

# app.mount("/model",StaticFiles(directory="model"), name="model")
# model = joblib.load(Path(".\\model\\prediction_model.pkl"))

model = joblib.load(os.path.join(os.path.dirname(__file__),'model','prediction_model.pkl'))
# Input request/payload structure
class DiamondFeatures(BaseModel):
    cut:int
    color:int
    clarity:int
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
        request.cut,
        request.color,
        request.clarity
    ]])

    y_predict = model.predict(matrix_feature)
    return y_predict[0]
