from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import pickle
np.random.seed(32113)

def data_preparer_ensemble(df1,df2,df3,df4, lbl = 'word', countries=['US','BR','RU','KR'],\
                   words=['cat','tiger','lion','dog'],sample=30000):
    '''
    Function:
    process dataframes so that it can be used for xgboost, random forest and other ensemble methods.
    the function prepares dataframe for image recognition model.
    
    Input:
    df1,2,3,4 = dataframes with different topics (cat,dog,lion,tiger) [dataframe]
    lbl = "word": word is used when running image recognition.

    words = list of string that contains words of topic of interest [list]
    sample = max number of data to take in (used when lbl = word)

    Output:
    new_df = dataframe or the non-label features of your model
    Y = a label feature of your model

    note: uses random.seed(32113)
    '''
    # running image recognition
    if lbl == 'word':
      #runs _df_initial_fixer for each word to prepare dataframe
        df_test1 = _df_initial_fixer(df1,words[0],sample)
        df_test2 = _df_initial_fixer(df2,words[1],sample)
        df_test3 = _df_initial_fixer(df3,words[2],sample)
        df_test4 = _df_initial_fixer(df4,words[3],sample)
        print(len(df_test1),len(df_test2),len(df_test3),len(df_test4))

        # convining all 4 dataframe to create a new dataframe. the new_df will be the input for XGB.
        new_df = pd.concat([df_test1,df_test2,df_test3,df_test4], axis =0)
        Y = new_df.pop('word')
        b_loon={}
        for i in xrange(len(words)):
            b_loon[words[i]] = i
        # Y will be the label for my XGB model.
        Y = Y.map(b_loon)
        return new_df,Y

def _df_initial_fixer(df, word, sample=60000):
    '''
    function:
    - ramdomly select rows (image) "sample" times from the df dataframe
    and delete features that are not used in ensemble method modeling

    input:
        df = dataframe. output of 1_feature_engineering_func. [pd.dataframe]
        word = name of topic ig "cat" [str]
        sample = number of sample you want to extract from df [int]

    output:
    new data frame!

    '''
    print "total number of images for df_{}: {}".format(word, len(df))
    random_index = np.random.choice(list(df.index), sample, replace=False)
    df = df.loc[list(random_index)]
    df_test = df.drop(['drawing','key_id','timestamp','recognized','X','Y','time',\
                        'X_per_stroke','Y_per_stroke','time_per_stroke',\
                        'total_time_of_stroke','dp_per_stroke','dp_percent_per_stroke',\
                        'direction'], axis=1)
    return df_test


#################### ipynb ##########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import vectorize
% matplotlib inline
import json
import time
np.random.seed(32113)
import feature_engineering_func as fe_func
import cnn_func
import ensemble_method_func as em_func
from sklearn.model_selection import train_test_split
import xgboost as XGB
from sklearn.model_selection import GridSearchCV

# data prep for ensemble method model
category = "cat_word_ems"
filepath = "./data/raw_data/full%2Fraw%2Fcat.ndjson"
df = pd.read_json(filepath, lines=True)

fe_func.feature_engineering_ensemble(df,category,60000,'word')

df_cat_en2 = pd.read_pickle('')
df_tiger_en2 = pd.read_pickle('')
df_lion_en2 = pd.read_pickle('')
df_dog_en2 = pd.read_pickle('')

X2,Y2 = em_func.data_preparer_ensemble(df_cat_en2,df_tiger_en2, df_lion_en2,df_dog_en2, \
                                          lbl = 'word', countries=['US','BR','RU','KR'],\
                                                     words=['cat','tiger','lion','dog'],\
                                                            sample=30000, limit = 8000)

X_tr4,X_te4,y_tr4,y_te4 =train_test_split(X2,Y2,test_size = 0.15, \
                                                    random_state=831713, stratify = Y2)
# GRID SEARCH for image recognition
xgb_test2 = XGB.XGBClassifier()
parameters ={'max_depth':[1], 'n_estimators':[5000],'learning_rate':[0.25,0.5]}
Gsearch2 = GridSearchCV(xgb_test2,parameters)

# Exhaustive search over specified parameter values for an estimator
start_time = time.time()
Gsearch2.fit(X_tr4,y_tr4)
print("--- %s seconds ---" % (time.time() - start_time))


# XGBoost image recognition
xgb2 =XGB.XGBClassifier(max_depth=1, n_estimators=5000, learning_rate=0.25)
xgb2.fit(np.array(X_tr4),np.array(y_tr4))
score = xgb2.score(np.array(X_te4),np.array(y_te4))
print "XGBoost image recognition accuracy: {} percent".format(score*100)
