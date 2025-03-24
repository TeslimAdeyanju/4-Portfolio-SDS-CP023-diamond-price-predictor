import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load('model/PricePredictor.pkl')
sc = joblib.load('model/StdScaler.pkl')
oe = joblib.load('model/OrdEncoder.pkl')

app = FastAPI(
    title = 'Diamond Price Predictor API',
    description = 'An API for predicting price of a diamond based on its various attributes using XGBoost',
    version = '1.0.0'
)

# Define the input data structure
class DiamondData(BaseModel):
    carat: float        
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

    @app.post(path = '/api/predict',
              summary = 'Predict the price of a diamond',
              description = 'Predicting the price of a diamond based on its various attributes using XGBoost',
              tags = ['Diamond Price Predictor']
        )
    def predict(input_data: DiamondData):
        """
            :param diamond_data: 
                carat - 
                cut - 
                color - 
                clarity - 
                depth - 
                table - 
                x - 
                y - 
                z - 
            :return:
                Predicted price of the diamond
            
        """

        # Convert the input data to a numpy array
        input_data = np.array([
                input_data.carat,
                input_data.cut,
                input_data.color,
                input_data.clarity,
                input_data.depth,
                input_data.table,
                input_data.x,
                input_data.y,
                input_data.z
            ]).reshape(1, -1)
        
        # Encode the categorical data
        encoded_data = oe.transform(input_data[:, 1:4])
        input_data[:, 1:4] = encoded_data

        # Scale the numerical data
        scaled_data = sc.transform(input_data)

        # Make the prediction
        prediction = model.predict(scaled_data)[0]

        return {'prediction': prediction}
    



