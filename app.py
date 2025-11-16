from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBRegressor
import json

# importing model
model = XGBRegressor()
model.load_model('model.json')

# Creating a input validation class
class Features(BaseModel):
    t_1: float
    t_2: float
    t_3: float
    t_4: float
    t_5: float
    t_6: float

# Create a FastAPI instance
app = FastAPI()

# Adding a get request so that vistng the application does not return an error
@app.get("/")
def root():
    return {'Hello, welcome to my Car prediction model'}
# Defining an endpoint
@app.post("/predict")
def predict(data: Features):
    test_data = [[
        data.t_1,
        data.t_2,
        data.t_3,
        data.t_4,
        data.t_5,
        data.t_6
    ]]
    sales = model.predict(test_data)
    return  float(sales)

