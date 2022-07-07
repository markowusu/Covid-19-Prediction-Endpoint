import datetime
import pickle
import pandas as pd
import numpy as np
import prophet
from typing import  Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import predict_confirmed, convert_confirmed,predict_death,convert_death

app = FastAPI()


app = FastAPI()

origins = [
    "http://localhost:3000",
  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# pydantic models


class Data_input(BaseModel):
    day: int




# routes


@app.get("/ping")
async def pong():
    return {"ping": "pong!"}


#Route For Confirmed Cases

@app.post("/confirmed/daily")
def get_prediction(days: Data_input):
    data = []
    day = days.day
    for x in range(day):
        prediction_list = predict_confirmed(x)  
        response_object =  convert_confirmed(prediction_list)
        print(response_object)
        data.append(response_object)
    return data
   

#Route For Death Cases

@app.post("/death/daily")
def get_prediction(days: Data_input):
    data = []
    day = days.day
    for x in range(day):  
        prediction_list = predict_death(day)  
        response_object =  convert_death(prediction_list)
        print(response_object)
        data.append(response_object)
    return data
   






