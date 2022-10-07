import nltk
import pandas as pd 
from nltk.corpus import stopwords
import numpy as np
import nltk 
from sklearn.base import TransformerMixin, BaseEstimator
from nltk.corpus import stopwords 
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text  import CountVectorizer 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb 

covid_dataframe = pd.read_csv("https://raw.githubusercontent.com/usmaann/COVIDSenti/main/COVIDSenti-A.csv")

nltk.download('punkt')
nltk.download('stopwords')


pstem = PorterStemmer()
lem = WordNetLemmatizer()
stop_words = stopwords.words('english')

covid_dataframe["cleaned_tweet"] = ""

def cleanDataset(rows=1,all_rows = False):
    cleaned_sentence = " "
    covid_dataframe_shape = covid_dataframe.shape
    if all_rows :
        
        rows = covid_dataframe_shape[0]
    else:
        print("The number of rows is %d" % (rows))
        
    print("Number of rows %2d you have to loop" % covid_dataframe_shape[0])
    for row in range(rows):
        cleaned_text = re.sub('https?://[A-Za-z0-9./]+','',covid_dataframe["tweet"][row])
#         print(covid_dataframe["tweet"][row])
        # remove hash syumbols from the dataset 
        cleaned_text = re.sub("[^a-zA-Z]"," ",cleaned_text)

        #removing RT tags in tweets
        cleaned_text = re.sub(r'^[RT]+',' ',cleaned_text)

        cleaned_text = re.sub("CDC","center disease contorl prevention", cleaned_text)
        cleaned_text = re.sub("WHO","world health organization", cleaned_text)



        #covert to lower case 
        cleaned_text = cleaned_text.lower()
        # removing stop words from the dataset 

        tokens = nltk.word_tokenize(cleaned_text)


#         cleaned_sentence.join((word for word in tokens if word not in stop_words)) 
        covid_dataframe['cleaned_tweet'][row] = str(cleaned_sentence.join((word for word in tokens if word not in stop_words)))
        # print(covid_dataframe['cleaned_tweet'][row],"cleaned text %2d" %row )
        cleaned_sentence = " "


    
    
    return cleaned_sentence.join((word for word in tokens if word not in stop_words)) 
    

cleanDataset(rows=300, all_rows=False)



class get_vector_transformation(TransformerMixin, BaseEstimator):

    def __init__(self, cleaned_data):
        self.cleaned_data = cleaned_data

    def fit(self, x, y =None):
        return self 

    def transform(self,x,y = None):  
        transform_data = np.array(self.cleaned_data.loc[:,"cleaned_tweet"]).ravel()
        vector = CountVectorizer()
        converted_data  = vector.fit_transform(transform_data).todense()
        return converted_data
    

covid_dataframe["label"].replace("neu",0,inplace=True)
covid_dataframe["label"].replace("neg",-1,inplace=True)
covid_dataframe["label"].replace("pos",1,inplace=True)

X= get_vector_transformation(covid_dataframe) # getting the column 
# X = countVector.transform(np.array(covid_dataframe.loc[:,"cleaned_tweet"]).ravel())   # vector space for text 
# labelCountVector = get_vector_transformation(np.array(data.loc[:,"label"]).ravel()) 
# Y = labelCountVector.transform(np.array(covid_dataframe.loc[:,"label"]).ravel()) # vector space for the label data
Y = np.array(covid_dataframe.iloc[:,1]).ravel()

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, train_size=0.8, random_state=42)
# # I used the random_state to be "42" because it will yeild constants factor of randomness and not madness
# # perfect number, {the answer to life, the universe and everything}. Hitchhiker's guide to life, the universe and everything 



# xgb_Classifier = xgb.XGBClassifier()
# xgb_Classifier.fit(X_train,Y_train)

# print("Dumping the file in a joblib file ")
# import joblib

# model_file = "sentiment_analysis.joblib"
# sentiment_analysis_file = joblib.dump(xgb_Classifier,model_file)


import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load some categories of newsgroups dataset
categories = [
    "soc.religion.christian",
    "talk.religion.misc",
    "comp.sys.mac.hardware",
    "sci.crypt",
]

sentiment_category = [ "positive","neutral","negative"]
newsgroups_training = fetch_20newsgroups(
    subset="train", categories=categories, random_state=0
)
newsgroups_testing = fetch_20newsgroups(
    subset="test", categories=categories, random_state=0
)

# Make the pipeline
model = make_pipeline(
    cleanDataset(rows=300),
    get_vector_transformation(),
    # TfidfVectorizer(),
    MultinomialNB(),
)

# Train the model
model.fit(X_train, Y_train)

# Serialize the model and the target names
model_file = "newsgroups_model.joblib"
model_targets_tuple = (model,sentiment_category )
joblib.dump(model_targets_tuple, model_file)