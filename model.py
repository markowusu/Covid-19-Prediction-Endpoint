import datetime
import pickle
import pandas as pd
import numpy as np
from fbprophet import Prophet
pd.set_option("display.float_format",lambda x: '%.f' % x)

TODAY = datetime.date.today()


#training our model
def train():
    df = pd.read_csv('covid_19_clean_complete_2022.csv')
    df.head(5)
    df  = df[["Date","Confirmed","Deaths","Recovered","Active"]]
    confirmed = df.groupby("Date").agg({'Confirmed':'sum'})
    confirmed = confirmed.reset_index()
    confirmed.columns = ['ds','y']
    confirmed['ds'] = pd.to_datetime(confirmed['ds'])
    model = Prophet()
    model.fit(confirmed)

    pickle.dump(model, open('model.pkl',"wb"))
    


#predicting outcome
def predict(days):
    model =pickle.load(open('model.pkl',"rb"))
    future =TODAY + datetime.timedelta(days=days)
    dates = pd.date_range(start=TODAY,end=future.strftime("%m/%d/%Y"),)
    df = pd.DataFrame({"ds":dates})
    forecast = model.predict(df)
    predict = forecast.tail(days+1).to_dict("records")
    return predict

#
def convert(prediction_list):
    output={}
    for df in prediction_list:
        pre_data = df["ds"].strftime("%m/%d/%Y")
        output[pre_data] = df["yhat"]
    return output
