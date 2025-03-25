import joblib
import numpy as np
import uvicorn
import os
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load('model/PricePredictor.pkl')
oe = joblib.load('model/OrdEncoder.pkl')

app = FastAPI(
    title = 'Diamond Price Predictor API',
    description = 'An API for predicting price of a diamond based on its various attributes using XGBoost',
    version = '1.0.0'
)

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Render!"}

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
    # Convert input data to dictionary
    input_dict = input_data.dict()

    # Convert the input data to a numpy array
    raw_data = np.array([
            
            input_dict["cut"],
            input_dict["color"],
            input_dict["clarity"],
            input_dict["carat"],
            input_dict["depth"],
            input_dict["table"],
            input_dict["x"],
            input_dict["z"],
            input_dict["y"]
        ]).reshape(1, -1)
    
    # Encode the categorical data
    encoded_data = oe.transform(raw_data[:, 0:3])
    raw_data[:, 0:3] = encoded_data

    # Convert entire array to float for XGBoost compatibility
    raw_data = raw_data.astype(float)

    # Make the prediction
    prediction = model.predict(raw_data)
    
    return {"prediction": float(prediction[0])}
    


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Get PORT from environment, default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)