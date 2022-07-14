import pandas as pd 
result_df = []
def getConfrimedcase():
    file_csv = pd.read_csv("/Users/markowusu/covid-project/Backend--ML-final-year/utils/covid_19_clean_complete_2022.csv")
    file_csv.drop(file_csv.columns[[0,1,2]], axis=1, inplace=True)
    print(file_csv)
    

if __name__ == '__main__':
    getConfrimedcase()