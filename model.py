import datetime
import pickle
import pandas as pd
import joblib
import numpy as np
import re 
import nltk 
from nltk.corpus import stopwords 
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text  import CountVectorizer 
from prophet import Prophet
pd.set_option("display.float_format",lambda x: '%.f' % x)

TODAY = datetime.date.today()

pstem = PorterStemmer()
lem = WordNetLemmatizer()
stop_words = stopwords.words('english')

#training our model
# def train():
#     df = pd.read_csv('covid_19_clean_complete_2022.csv')
#     df.head(5)
#     df  = df[["Date","Confirmed","Deaths","Recovered","Active"]]
#     confirmed = df.groupby("Date").agg({'Confirmed':'sum'})
#     confirmed = confirmed.reset_index()
#     confirmed.columns = ['ds','y']
#     confirmed['ds'] = pd.to_datetime(confirmed['ds'])
#     model = Prophet()
#     model.fit(confirmed)

#     pickle.dump(model, open('model.pkl',"wb"))
    


#predicting Confirmed Cases
def predict_confirmed(days):
    model =pickle.load(open('confirmed_model.pkl',"rb"))
    future =TODAY + datetime.timedelta(days=days)
    dates = pd.date_range(start=TODAY,end=future.strftime("%m/%d/%Y"),)
    df = pd.DataFrame({"ds":dates})
    forecast = model.predict(df)
    predict = forecast.tail(days+1).to_dict("records")
    return predict


def convert_confirmed(prediction_list):
    output={}
    for df in prediction_list:
        date = df["ds"].strftime("%m/%d/%Y")
        output[date] = df["yhat"] + 54000000
    return output


#predicting Death Cases
def predict_death(days):
    model =pickle.load(open('death_model.pkl',"rb"))
    future =TODAY + datetime.timedelta(days=days)
    dates = pd.date_range(start=TODAY,end=future.strftime("%m/%d/%Y"),)
    df = pd.DataFrame({"ds":dates})
    forecast = model.predict(df)
    predict = forecast.tail(days+1).to_dict("records")
    return predict


def convert_death(prediction_list):
    output={}
    for df in prediction_list:      
        date = df["ds"].strftime("%m/%d/%Y")
        output[date] = df["yhat"]  
    return output


#  TODO: create a pipeline for the model and store in the cache in a Joblib format 
def sentiment_analysis(data):
    model = joblib.load(open('utils/sentiment_analysis.joblib','rb'))
    
    

    cleaned_sentence = " "
    cleaned_text =  re.sub('https?://[A-Za-z0-9./]+','',data)
    # remove hash symbols from the dataset 
    cleaned_text = re.sub("[^a-zA-Z]"," ",cleaned_text)

            #removing RT tags in tweets
    cleaned_text = re.sub(r'^[RT]+',' ',cleaned_text)

    cleaned_text = re.sub("CDC","center disease control prevention", cleaned_text)
    cleaned_text = re.sub("WHO","world health organization", cleaned_text)



            #covert to lower case 
    cleaned_text = cleaned_text.lower()
            # removing stop words from the dataset 

    tokens = nltk.word_tokenize(cleaned_text)

    data_output = cleaned_sentence.join((word for word in tokens if word not in stop_words))
    # print(data_output)

    # cleaned_sentence = " "

    vector = CountVectorizer()
    converted_data  = vector.fit_transform(np.array(data_output).ravel()).toarray()
    print(converted_data,"element")
  # dispaly(converted_data,"This is the data to be vectorized")
    p = model.predict(converted_data)
    target = ["neutral","positive","negative"]    
    # print(target[p[0]],"amazing")

    return {"message":p}
        