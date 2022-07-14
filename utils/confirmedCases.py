import pandas as pd 

def getConfrimedcase():
    file_csv = pd.read_csv("/utils/covid_19_clean_complete_2022.csv")
    print(file_csv)
