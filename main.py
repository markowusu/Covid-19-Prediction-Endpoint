import datetime
import pickle
import pandas as pd
import numpy as np
from fbprophet import Prophet
from typing import  Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import convert, predict

app = FastAPI()


# pydantic models


class Data_input(BaseModel):
    day: int




# routes


@app.get("/ping")
async def pong():
    return {"ping": "pong!"}


@app.post("/predict/")
def get_prediction(days: Data_input):
    day = days.day
    prediction_list = predict(day)  
    response_object = {"ticker": days, "forecast": convert(prediction_list)}
    return response_object






# @app.get("/")
# def read_root():
#     return{"hello":"world"}




# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

# @app.get('/uploads')
# async def upload(csv_file:UploadFile = File(...)):  
#     contents = await file.read()
#     buffer = BytesIO(contents)
#     df = pd.read_csv(buffer)
#     buffer.close()
#     print(df.head(5))
#     return df.to_dict(orient='records')
   


