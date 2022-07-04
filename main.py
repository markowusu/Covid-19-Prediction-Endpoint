import datetime
import pickle
import pandas as pd
import numpy as np
from fbprophet import Prophet
from typing import  Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import predict_confirmed, convert_confirmed,predict_death,convert_death

app = FastAPI()


# pydantic models


class Data_input(BaseModel):
    day: int




# routes


@app.get("/ping")
async def pong():
    return {"ping": "pong!"}


#Route For Confirmed Cases

@app.post("/confirmed/")
def get_prediction(days: Data_input):
    day = days.day
    prediction_list = predict_confirmed(day)  
    response_object = {"ticker": days, "forecast": convert_confirmed(prediction_list)}
    return response_object
   

#Route For Death Cases

@app.post("/death/")
def get_prediction(days: Data_input):
    day = days.day
    prediction_list = predict_death(day)  
    response_object = {"ticker": days, "forecast": convert_death(prediction_list)}
    return response_object
   




