import re 

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
    