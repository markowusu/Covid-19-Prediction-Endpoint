import pandas as pd 
result_cases= []
result_labels= []
def getConfrimedcase():
    file_csv = pd.read_csv("/Users/markowusu/covid-project/Backend--ML-final-year/utils/confirmed.csv")
    
    for _, row in file_csv.iterrows():
        result_cases.append(row["Date"])
        result_labels.append(row["Confirmed"])
    print(result_cases)

if __name__ == '__main__':
    getConfrimedcase()