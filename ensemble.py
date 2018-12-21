
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import vectorize
import json
import time
np.random.seed(32113)

##############################################################################
#                             Aggregated functions                           #
##############################################################################


def feature_engineering_ensemble(df,category,sample=200,purpose='word',                                            countries = ['US','BR','RU','KR']):

    start_time = time.time()
    #runs feature_eng_pt1 through pt5.
    df = df[:3000]
    df_test1 = feature_eng_pt1(df)
    df_test2 = feature_eng_pt2(df_test1)
    df_test3 = feature_eng_pt3(df_test2)
    df_subset = feature_eng_pt4(df_test3)
    df_subset2 = feature_eng_pt5(df_test3)
    df_final = pd.concat([df_test3,df_subset,df_subset2], axis=1)
    
    # prepares final dataframe
    #If purpose = 'word' it will randomly select 'sample' number of datapoints from df_final
    # if purpose == 'word':
    #     df_final.index = range(len(df_final))
    #     random_ind = np.random.choice(list(df_final.index), sample, replace=False)
    #     df_final = df_final.loc[list(random_ind)]
    df_final.to_pickle("./data/MY_feature_{}.pkl".format(category))
    print("--- %s seconds ---" % (time.time() - start_time))


def feature_eng_pt1(df_cf):
    # create feature "stroke_number"
    df_cf['stroke_number']=df_cf['drawing'].str.len()

    #create feature "final_time"
    df_cf['final_time'] = [df_cf.loc[index,'drawing']                [df_cf.stroke_number[index]-1][2][-1] for index in df_cf.index]

    #setting boolean and changing recognized features to 1 and 0.
    b_loon = {True: 1, False:0}
    df_cf['recognized'] = df_cf['recognized'].map(b_loon)

    #filtered data by stroke number, recognized and final time features
    df_cf = df_cf[(df_cf['recognized']==1) & (df_cf['stroke_number'] <= 15)]
    df_cf = df_cf[(df_cf['final_time']<=20000)]
    return df_cf


def feature_eng_pt2(df_cf):
    X = {}
    Y = {}
    Xperst = {}
    Yperst = {}
    Ymax ={}
    time = {}
    tperst = {}
    Tdiff = {}
    ttnum_dp = {}
    Tdiffmax = {}
    Tdiffmin = {}
    Tdiffstd = {}
    dpps = {}
    dppps = {}
    dp_max = {}
    dp_min = {}
    dp_std = {}
    sumtimeps = {}

    for i in df_cf.index:
        num = df_cf.stroke_number[i]
        #store X,Y,time of the stroke in a temp list
        Xt = [df_cf.loc[i,'drawing'][stroke][0] for stroke in range(num)]
        Yt = [df_cf.loc[i,'drawing'][stroke][1] for stroke in range(num)]
        tt = [df_cf.loc[i,'drawing'][stroke][2] for stroke in range(num)]
        # calculate the difference between final and initial time of a stroke
        Tdifftemp = [(df_cf.loc[i,'drawing'][stroke][2][-1] - df_cf.loc[i,'drawing'][stroke][2][0])                     for stroke in range(num)]
        # calculate the length of the stroke list
        dpps_temp = [len(df_cf.loc[i,'drawing'][stroke][2]) for stroke in range(num)]

        #store all X(or Y or time) info of an image into a list
        Xtemp = [item for stroke in Xt for item in stroke]
        Ytemp = [item for stroke in Yt for item in stroke]
        time[i] = [item for stroke in tt for item in stroke]

        #normalizing X and Y
        Xmintemp = np.min(Xtemp)-1
        Xmaxtemp = np.max(Xtemp)+1
        Ymintemp = np.min(Ytemp)-1
        #runs user defined function array_normalizer to normalize
        Xnorm = _array_normalizer(Xtemp, Xmintemp,Xmaxtemp,Xmintemp)
        Ynorm = _array_normalizer(Ytemp, Xmintemp,Xmaxtemp,Ymintemp)
        Ymax[i] = np.max(Ynorm)
        X[i] = Xnorm
        Y[i] = Ynorm
        #store X,Y and time info from each stroke as a list
        Xperst[i] = [list(_array_normalizer(Xt[stroke],Xmintemp,Xmaxtemp,Xmintemp)) for stroke in range(len(Xt))]
        Yperst[i] = [list(_array_normalizer(Yt[stroke],Xmintemp,Xmaxtemp,Ymintemp)) for stroke in range(len(Yt))]
        tperst[i] = [tt[stroke] for stroke in range(len(tt))]
        
        #total number of datapoints 
        ttnum_dp[i] = len(Xnorm)
        
        #store time spent on each stroke
        Tdiff[i] = Tdifftemp
        #store index of stroke that user spent most time
        Tdiffmax[i] = np.argmax(Tdifftemp)
        #store index of stroke that user spent least time
        Tdiffmin[i] = np.argmin(Tdifftemp)
        #time standard deviation for each stroke
        Tdiffstd[i] = np.std(Tdifftemp)
        
        #number of datapoints for each stroke
        dpps[i] = dpps_temp
        #number of datapoints stored as a percentage
        dppps[i] = np.array(dpps_temp)/float(len(Xtemp))
        #stroke with maximum number of datapoints
        dp_max[i] = np.argmax(dpps_temp)
        #stroke with minimum number of datapoints
        dp_min[i] = np.argmin(dpps_temp)
        #std. of datapoints per stroke
        dp_std[i] = np.std(dpps_temp)
        #total time spent on drawing
        sumtimeps[i] = sum(Tdifftemp)
        
    # create new features
    df_cf['total_number_of_datapoints'] = pd.Series(ttnum_dp)
    df_cf['X'] = pd.Series(X)
    df_cf['Y'] = pd.Series(Y)
    df_cf['Ymax'] = pd.Series(Ymax)
    df_cf['time'] = pd.Series(time)
    df_cf['total_time_of_stroke'] = pd.Series(Tdiff)
    df_cf['dp_per_stroke'] = pd.Series(dpps)
    df_cf['dp_percent_per_stroke'] = pd.Series(dppps)
    df_cf['stroke_with_max_time'] = pd.Series(Tdiffmax)
    df_cf['stroke_with_min_time'] = pd.Series(Tdiffmin)
    df_cf['std_of_time'] = pd.Series(Tdiffstd)
    df_cf['ave_datapoints_per_stroke'] = df_cf['total_number_of_datapoints']/(df_cf['stroke_number'])
    df_cf['total_time_drawing'] = pd.Series(sumtimeps)
    df_cf['ave_time_per_stroke'] = df_cf['total_time_drawing']/(df_cf['stroke_number'])
    df_cf['stroke_with_max_dp'] = pd.Series(dp_max)
    df_cf['stroke_with_min_dp'] = pd.Series(dp_min)
    df_cf['X_per_stroke'] = pd.Series(Xperst)
    df_cf['Y_per_stroke'] = pd.Series(Yperst)
    df_cf['time_per_stroke'] = pd.Series(tperst)
    df_cf['std_of_dp'] = pd.Series(dp_std)
    df_cf = df_cf[df_cf['Ymax']<=1.5]
    return df_cf

def feature_eng_pt3(df_cf):
    direction = {}
    for index in df_cf.index:
        dx = [float(df_cf.drawing[index][stroke][1][-1] - df_cf.drawing[index][stroke][1][0])           for stroke in range(df_cf.stroke_number[index])]
        dy = [float(df_cf.drawing[index][stroke][0][-1] - df_cf.drawing[index][stroke][0][0])           for stroke in range(df_cf.stroke_number[index])]
        dx = np.array(dx)
        dy = np.array(dy)
        dx[dx==0] = 0.000001
        vecrad_direction = np.vectorize(_radian_direction)
        direction[index] = vecrad_direction(dy,dx)
    df_cf['direction'] = pd.Series(direction)
    return df_cf

def feature_eng_pt4(df_cf):
    ar = np.zeros((len(df_cf),75))
    c = 0
    for index_ in df_cf.index:
        stroke = (df_cf.stroke_number[index_])
        ar[c][:stroke] = np.array(df_cf['dp_percent_per_stroke'][index_])
        ar[c][15:15+stroke] = np.array(df_cf['direction'][index_])
        ar[c][30:30+stroke] = np.array(df_cf['total_time_of_stroke'][index_])
        ar[c][45:45+stroke] = np.array(df_cf['dp_per_stroke'][index_])
        ar[c][60:75] = np.array([0]*stroke+[1]*(15-stroke))
        c += 1
    subset = pd.DataFrame(ar)
    subset.index = df_cf.index
    for num in range(15):
        subset = subset.rename(columns={num:"datapoint_percentage_stroke{}".format(num)})
    for num in range(15,30):
        subset = subset.rename(columns={num:"direction_stroke{}".format(num-15)})
    for num in range(30,45):
        subset = subset.rename(columns={num:"time_stroke{}".format(num-30)})
    for num in range(45,60):
        subset = subset.rename(columns={num:"datapoint_stroke{}".format(num-45)})
    for num in range(60,75):
        subset = subset.rename(columns={num:"switch_stroke{}".format(num-60)})
    return subset

def feature_eng_pt5(df_cf):
   ar = np.zeros((len(df_cf),300))
    c = 0
    for index_ in df_cf.index:
        Xpoints = [_value_from_stroke(df_cf['dp_per_stroke'][index_][stroke],                                    df_cf['dp_percent_per_stroke'][index_][stroke],                                    df_cf['X_per_stroke'][index_][stroke])                                    for stroke in range(df_cf.stroke_number[index_])]

        Ypoints = [_value_from_stroke(df_cf['dp_per_stroke'][index_][stroke],                                    df_cf['dp_percent_per_stroke'][index_][stroke],                                    df_cf['Y_per_stroke'][index_][stroke])                                    for stroke in range(df_cf.stroke_number[index_])]

        tpoints = [_value_from_stroke(df_cf['dp_per_stroke'][index_][stroke],                                    df_cf['dp_percent_per_stroke'][index_][stroke],                                    df_cf['time_per_stroke'][index_][stroke])                                    for stroke in range(df_cf.stroke_number[index_])]

        X = [item for stroke in Xpoints for item in stroke]
        Y = [item for stroke in Ypoints for item in stroke]
        time = [item for stroke in tpoints for item in stroke]

        #if the number datapoints turn out to be less than 100, it will fill
        #empty cell with it's last data points.
        if len(X)<100:
            X = X + [X[-1]]*(100-len(X))
        if len(Y)<100:
            Y = Y + [Y[-1]]*(100-len(Y))
        if len(time)<100:
            time = time + [time[-1]]*(100-len(time))

        ar[c][:100] = np.array(X[0:100])
        ar[c][100:200] = np.array(Y[0:100])
        ar[c][200:300] = np.array(time[0:100])
        c += 1

    subset = pd.DataFrame(ar)
    subset.index = df_cf.index
    for num in range(100):
        subset = subset.rename(columns={num:"X_{}".format(num)})
    for num2 in range(100,200):
        subset = subset.rename(columns={num2:"Y_{}".format(num2-100)})
    for num3 in range(200,300):
        subset = subset.rename(columns={num3:"time_{}".format(num3-200)})
    return subset

def _array_normalizer(array1,Xmin,Xmax,array_min):
    return (np.array(array1)-np.array([array_min]*len(array1)))/float(Xmax-Xmin)

def _radian_direction(dy,dx):
    if dy < 0.0 and dx > 0.0:
        return (2*np.pi + np.arctan(dy/dx))
    elif dy >=0.0 and dx > 0.0:
        return (np.arctan(dy/dx))
    else:
        return np.pi + np.arctan(dy/dx)


def _value_from_stroke(stroke_length,percentage,xperstroke):
    idxs = np.around(np.linspace(0,stroke_length-1,int(np.around(percentage*100))))
    return [xperstroke[int(ind)] for ind in idxs]

def load_json(filename):

    df = pd.read_json(filename, lines=True)
    test = df.groupby(df['countrycode']).count()
    print(test.sort(columns='drawing',ascending=False).head(15))
    return df

def pic_viewer(df_cf, _id):
    plt.scatter(df_cf.X[_id],df_cf.Y[_id])
    plt.gca().invert_yaxis()


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import vectorize
get_ipython().run_line_magic('matplotlib', 'inline')
import json
import time
import feature_engineering_func as fe_func
#import cnn_func
import ensemble_method_func as em_func
from sklearn.model_selection import train_test_split
import xgboost as XGB
from sklearn.model_selection import GridSearchCV


# ## data prep for ensemble method model

# In[3]:


category = "lion"
filepath = "lion.ndjson"
df = pd.read_json(filepath, lines=True)


# In[10]:


feature_engineering_ensemble(df,category,'word')


# In[11]:


category = "tiger"
filepath = "tiger.ndjson"
df = pd.read_json(filepath, lines=True)
feature_engineering_ensemble(df,category,'word')


# In[12]:


category = "cat"
filepath = "cat.ndjson"
df = pd.read_json(filepath, lines=True)
feature_engineering_ensemble(df,category,'word')


# In[92]:


category = "bush"
filepath = "bush.ndjson"
df = pd.read_json(filepath, lines=True)
feature_engineering_ensemble(df,category,'word')


# In[93]:


category = "grass"
filepath = "grass.ndjson"
df = pd.read_json(filepath, lines=True)
feature_engineering_ensemble(df,category,'word')


# In[94]:


category = "house plant"
filepath = "house plant.ndjson"
df = pd.read_json(filepath, lines=True)
feature_engineering_ensemble(df,category,'word')


# In[95]:


category = "computer"
filepath = "computer.ndjson"
df = pd.read_json(filepath, lines=True)
feature_engineering_ensemble(df,category,'word')


# In[96]:


category = "camera"
filepath = "camera.ndjson"
df = pd.read_json(filepath, lines=True)
feature_engineering_ensemble(df,category,'word')


# In[13]:


category = "dog"
filepath = "dog.ndjson"
df = pd.read_json(filepath, lines=True)
feature_engineering_ensemble(df,category,'word')


# In[34]:


category = "car"
filepath = "car.ndjson"
df = pd.read_json(filepath, lines=True)
feature_engineering_ensemble(df,category,'word')
category = "tree"
filepath = "tree.ndjson"
df = pd.read_json(filepath, lines=True)
feature_engineering_ensemble(df,category,'word')
category = "flower"
filepath = "flower.ndjson"
df = pd.read_json(filepath, lines=True)
feature_engineering_ensemble(df,category,'word')


# In[35]:


category = "train"
filepath = "train.ndjson"
df = pd.read_json(filepath, lines=True)
feature_engineering_ensemble(df,category,'word')
category = "mouse"
filepath = "mouse.ndjson"
df = pd.read_json(filepath, lines=True)
feature_engineering_ensemble(df,category,'word')
category = "school bus"
filepath = "school bus.ndjson"
df = pd.read_json(filepath, lines=True)
feature_engineering_ensemble(df,category,'word')
category = "duck"
filepath = "duck.ndjson"
df = pd.read_json(filepath, lines=True)
feature_engineering_ensemble(df,category,'word')


# In[38]:


category = "cow"
filepath = "cow.ndjson"
df = pd.read_json(filepath, lines=True)
feature_engineering_ensemble(df,category,'word')


# In[9]:


fe_func.feature_engineering_ensemble(df,category,3000,'word')

# In[58]:


df_cat_en = pd.read_pickle('./data/MY_feature_cat.pkl')
df_tiger_en = pd.read_pickle('./data/MY_feature_tiger.pkl')
df_lion_en = pd.read_pickle('./data/MY_feature_lion.pkl')
df_dog_en = pd.read_pickle('./data/MY_feature_dog.pkl')


# In[68]:


new_df,Y,df_US,df_BR,df_RU,df_KR = data_preparer_ensemble(df_cat_en2,df_tiger_en2,                                            df_lion_en2,df_dog_en2, lbl = 'countrycode',                                                         countries=['US','BR','RU','KR'],                                    words=['cat','tiger','lion','dog'],sample=1900, limit = 5000)


# In[69]:


X_tr3,X_te3,y_tr3,y_te3 =train_test_split(new_df,Y,test_size = 0.15,                                                     random_state=831713, stratify = Y)


# In[71]:


xgb =XGB.XGBClassifier(max_depth=1, n_estimators=1000, learning_rate=0.2)
xgb.fit(np.array(X_tr3),np.array(y_tr3))
score = xgb.score(np.array(X_te3),np.array(y_te3))
print("XGBoost Country prediction accuracy: {} percent".format(score*100))


# In[72]:


xgb =XGB.XGBClassifier(max_depth=1, n_estimators=5000, learning_rate=0.2)
xgb.fit(np.array(X_tr3),np.array(y_tr3))
score = xgb.score(np.array(X_te3),np.array(y_te3))
print("XGBoost Country prediction accuracy: {} percent".format(score*100))


# In[74]:


ind = np.argsort(xgb.feature_importances_)
imp = [X_te3.columns[i] for i in ind]
# top 15 important features
print(imp[-15:])


# In[76]:


new_df,Y,df_US,df_BR,df_RU,df_KR = data_preparer_ensemble(df_cat_en2, df_duck_en2, df_tree_en2,df_car_en2, lbl = 'countrycode',                                                         countries=['cat','duck','tree','car'],                                    words=['cat','tiger','lion','dog'],sample=1900, limit = 5000)
X_tr3,X_te3,y_tr3,y_te3 =train_test_split(new_df,Y,test_size = 0.15,                                                     random_state=831713, stratify = Y)
xgb =XGB.XGBClassifier(max_depth=1, n_estimators=5000, learning_rate=0.2)
xgb.fit(np.array(X_tr3),np.array(y_tr3))
score = xgb.score(np.array(X_te3),np.array(y_te3))
print("XGBoost Country prediction accuracy: {} percent".format(score*100))
ind2 = np.argsort(xgb.feature_importances_)
imp2 = [X_te4.columns[i] for i in ind2]
# top 15 important features
print(imp2[-15:])


# In[77]:


new_df,Y,df_US,df_BR,df_RU,df_KR = data_preparer_ensemble(df_cow_en2, df_tiger_en2, df_mouse_en2,df_school_bus_en2, lbl = 'countrycode',                                                         countries=['US','BR','RU','KR'],                                    words=['cow','tiger','mouse','school bus'],sample=1900, limit = 5000)
X_tr3,X_te3,y_tr3,y_te3 =train_test_split(new_df,Y,test_size = 0.15,                                                     random_state=831713, stratify = Y)
xgb =XGB.XGBClassifier(max_depth=1, n_estimators=1000, learning_rate=0.2)
xgb.fit(np.array(X_tr3),np.array(y_tr3))
score = xgb.score(np.array(X_te3),np.array(y_te3))
print("XGBoost Country prediction accuracy: {} percent".format(score*100))
ind = np.argsort(xgb.feature_importances_)
imp = [X_te3.columns[i] for i in ind]
# top 15 important features
print(imp[-15:])


# In[86]:


new_df,Y,df_US,df_BR,df_RU,df_KR = data_preparer_ensemble(df_cow_en2, df_tiger_en2, df_mouse_en2,df_school_bus_en2, lbl = 'countrycode',                                                         countries=['US','CN','JP','CA'],                                    words=['cow','tiger','mouse','school bus'],sample=1900, limit = 5000)
X_tr3,X_te3,y_tr3,y_te3 =train_test_split(new_df,Y,test_size = 0.15)
xgb =XGB.XGBClassifier(max_depth=1, n_estimators=1000, learning_rate=0.2)
xgb.fit(np.array(X_tr3),np.array(y_tr3))
score = xgb.score(np.array(X_te3),np.array(y_te3))
print("XGBoost Country prediction accuracy: {} percent".format(score*100))
ind = np.argsort(xgb.feature_importances_)
imp = [X_te3.columns[i] for i in ind]
# top 15 important features
print(imp[-15:])


# In[81]:


new_df,Y,df_US,df_BR,df_RU,df_KR = data_preparer_ensemble(df_cow_en2, df_tiger_en2, df_mouse_en2,df_school_bus_en2, lbl = 'countrycode',                                                         countries=['US','CN','IN','ID'],                                    words=['cow','tiger','mouse','school bus'],sample=1900, limit = 5000)
X_tr3,X_te3,y_tr3,y_te3 =train_test_split(new_df,Y,test_size = 0.15)
xgb =XGB.XGBClassifier(max_depth=1, n_estimators=1000, learning_rate=0.2)
xgb.fit(np.array(X_tr3),np.array(y_tr3))
score = xgb.score(np.array(X_te3),np.array(y_te3))
print("XGBoost Country prediction accuracy: {} percent".format(score*100))
ind = np.argsort(xgb.feature_importances_)
imp = [X_te3.columns[i] for i in ind]
# top 15 important features
print(imp[-15:])


# In[85]:


new_df,Y,df_1,df_2,df_3,df_4 = data_preparer_ensemble(df_cow_en2, df_tiger_en2, df_mouse_en2,df_school_bus_en2, lbl = 'countrycode',                                                         countries=['US','CN','IN','ID'],                                    words=['cow','tiger','mouse','school bus'],sample=1900, limit = 5000)
X_tr3,X_te3,y_tr3,y_te3 =train_test_split(new_df,Y,test_size = 0.15)
xgb =XGB.XGBClassifier(max_depth=1, n_estimators=1000, learning_rate=0.2)
xgb.fit(np.array(X_tr3),np.array(y_tr3))
score = xgb.score(np.array(X_te3),np.array(y_te3))
print("XGBoost Country prediction accuracy: {} percent".format(score*100))
ind = np.argsort(xgb.feature_importances_)
imp = [X_te3.columns[i] for i in ind]
# top 15 important features
print(imp[-15:])


# In[87]:


new_df,Y,df_1,df_2,df_3,df_4 = data_preparer_ensemble(df_cow_en2, df_tiger_en2, df_mouse_en2,df_school_bus_en2, lbl = 'countrycode',                                                         countries=['CA','GB','US','RU'],                                    words=['cow','tiger','mouse','school bus'],sample=1900, limit = 5000)
X_tr3,X_te3,y_tr3,y_te3 =train_test_split(new_df,Y,test_size = 0.15)
xgb =XGB.XGBClassifier(max_depth=1, n_estimators=1000, learning_rate=0.2)
xgb.fit(np.array(X_tr3),np.array(y_tr3))
score = xgb.score(np.array(X_te3),np.array(y_te3))
print("XGBoost Country prediction accuracy: {} percent".format(score*100))
ind = np.argsort(xgb.feature_importances_)
imp = [X_te3.columns[i] for i in ind]
# top 15 important features
print(imp[-15:])


# In[88]:


new_df,Y,df_1,df_2,df_3,df_4 = data_preparer_ensemble(df_cow_en2, df_tiger_en2, df_mouse_en2,df_school_bus_en2, lbl = 'countrycode',                                                         countries=['CZ','TH','DE','AU'],                                    words=['cow','tiger','mouse','school bus'],sample=1900, limit = 5000)
X_tr3,X_te3,y_tr3,y_te3 =train_test_split(new_df,Y,test_size = 0.15)
xgb =XGB.XGBClassifier(max_depth=1, n_estimators=1000, learning_rate=0.2)
xgb.fit(np.array(X_tr3),np.array(y_tr3))
score = xgb.score(np.array(X_te3),np.array(y_te3))
print("XGBoost Country prediction accuracy: {} percent".format(score*100))
ind = np.argsort(xgb.feature_importances_)
imp = [X_te3.columns[i] for i in ind]
# top 15 important features
print(imp[-15:])


# In[89]:


new_df,Y,df_1,df_2,df_3,df_4 = data_preparer_ensemble(df_cow_en2, df_tiger_en2, df_mouse_en2,df_school_bus_en2, lbl = 'countrycode',                                                         countries=['FI','SE','SA','IT'],                                    words=['cow','tiger','mouse','school bus'],sample=1900, limit = 5000)
X_tr3,X_te3,y_tr3,y_te3 =train_test_split(new_df,Y,test_size = 0.15)
xgb =XGB.XGBClassifier(max_depth=1, n_estimators=1000, learning_rate=0.2)
xgb.fit(np.array(X_tr3),np.array(y_tr3))
score = xgb.score(np.array(X_te3),np.array(y_te3))
print("XGBoost Country prediction accuracy: {} percent".format(score*100))
ind = np.argsort(xgb.feature_importances_)
imp = [X_te3.columns[i] for i in ind]
# top 15 important features
print(imp[-15:])


# In[90]:


new_df,Y,df_1,df_2,df_3,df_4 = data_preparer_ensemble(df_cow_en2, df_tiger_en2, df_mouse_en2,df_school_bus_en2, lbl = 'countrycode',                                                         countries=['DE','GB','US','CA'],                                    words=['cow','tiger','mouse','school bus'],sample=1900, limit = 5000)
X_tr3,X_te3,y_tr3,y_te3 =train_test_split(new_df,Y,test_size = 0.15)
xgb =XGB.XGBClassifier(max_depth=1, n_estimators=1000, learning_rate=0.2)
xgb.fit(np.array(X_tr3),np.array(y_tr3))
score = xgb.score(np.array(X_te3),np.array(y_te3))
print("XGBoost Country prediction accuracy: {} percent".format(score*100))
ind = np.argsort(xgb.feature_importances_)
imp = [X_te3.columns[i] for i in ind]
# top 15 important features
print(imp[-15:])


# In[91]:


new_df,Y,df_1,df_2,df_3,df_4 = data_preparer_ensemble(df_dog_en2, df_car_en2, df_mouse_en2,df_school_bus_en2, lbl = 'countrycode',                                                         countries=['DE','GB','US','CA'],                                    words=['dog','car','mouse','school bus'],sample=2500, limit = 5000)
X_tr3,X_te3,y_tr3,y_te3 =train_test_split(new_df,Y,test_size = 0.15)
xgb =XGB.XGBClassifier(max_depth=1, n_estimators=1000, learning_rate=0.2)
xgb.fit(np.array(X_tr3),np.array(y_tr3))
score = xgb.score(np.array(X_te3),np.array(y_te3))
print("XGBoost Country prediction accuracy: {} percent".format(score*100))
ind = np.argsort(xgb.feature_importances_)
imp = [X_te3.columns[i] for i in ind]
# top 15 important features
print(imp[-15:])


# In[ ]:





# #### GRID SEARCH for country prediction

# In[23]:


xgb_test = XGB.XGBClassifier()
parameters ={'max_depth':[1], 'n_estimators':[1000,1500,2000],'learning_rate':[0.1,0.2]}
Gsearch = GridSearchCV(xgb,parameters)


# In[24]:


start_time = time.time()
Gsearch.fit(X_tr3,y_tr3)
print("--- %s seconds ---" % (time.time() - start_time))


# In[25]:


Gsearch.best_estimator_


# In[52]:


ind = np.argsort(xgb.feature_importances_)


# In[54]:


imp = [X_te3.columns[i] for i in ind]


# ##### Top 15 important features for country prediction

# In[58]:


imp[-15:]


# ### Ensemble method Image Recognition

# In[29]:


df_cat_en2 = pd.read_pickle('./data/MY_feature_cat.pkl')
df_tiger_en2 = pd.read_pickle('./data/MY_feature_tiger.pkl')
df_lion_en2 = pd.read_pickle('./data/MY_feature_lion.pkl')
df_dog_en2 = pd.read_pickle('./data/MY_feature_dog.pkl')
X2,Y2 = data_preparer_ensemble(df_cat_en2,df_tiger_en2, df_lion_en2,df_dog_en2,                                           lbl = 'word', countries=['US','BR','RU','KR'],                                                     words=['cat','tiger','lion','dog'],                                                            sample = 1900)
X_tr4,X_te4,y_tr4,y_te4 =train_test_split(X2,Y2,test_size = 0.15,                                                     random_state=831713, stratify = Y2)


# In[43]:


X2,Y2 = data_preparer_ensemble(df_cat_en2,df_tiger_en2, df_lion_en2,df_dog_en2,                                           lbl = 'word', countries=['US','BR','RU','KR'],                                                     words=['cow','tiger','car','dog'],                                                            sample = 1900)
X_tr4,X_te4,y_tr4,y_te4 =train_test_split(X2,Y2,test_size = 0.15,                                                     random_state=831713, stratify = Y2)
print(y_tr4)


# #### GRID SEARCH for image recognition

# In[5]:


xgb_test2 = XGB.XGBClassifier()
parameters ={'max_depth':[1], 'n_estimators':[5000],'learning_rate':[0.25,0.5]}
Gsearch2 = GridSearchCV(xgb_test2,parameters)


# In[ ]:


start_time = time.time()
Gsearch2.fit(X_tr4,y_tr4)
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:


Gsearch2.best_estimator_


# #### XGBoost  image recognition

# In[33]:


xgb2 =XGB.XGBClassifier(max_depth=1, n_estimators=5000, learning_rate=0.25)
print(time.time())
X_tr4 = np.array(X_tr4)
y_tr4 = np.array(y_tr4)
print(X_tr4)
print(y_tr4)
xgb2.fit(X_tr4[:1000], y_tr4[:1000])
print(time.time())
score = xgb2.score(np.array(X_te4),np.array(y_te4))
print("XGBoost image recognition accuracy: {} percent".format(score*100))
# Feature importance for image recognition mode
ind2 = np.argsort(xgb2.feature_importances_)
imp2 = [X_te4.columns[i] for i in ind2]
# top 15 important features
print(imp2[-15:])


# In[39]:


df_cow_en2 = pd.read_pickle('./data/MY_feature_cow.pkl')
df_car_en2 = pd.read_pickle('./data/MY_feature_car.pkl')
df_tree_en2 = pd.read_pickle('./data/MY_feature_tree.pkl')
df_mouse_en2 = pd.read_pickle('./data/MY_feature_mouse.pkl')
df_train_en2 = pd.read_pickle('./data/MY_feature_train.pkl')
df_duck_en2 = pd.read_pickle('./data/MY_feature_duck.pkl')
df_flower_en2 = pd.read_pickle('./data/MY_feature_flower.pkl')
df_school_bus_en2 = pd.read_pickle('./data/MY_feature_school bus.pkl')


# In[50]:


X2,Y2 = data_preparer_ensemble(df_cow_en2,df_tiger_en2, df_car_en2,df_dog_en2,                                           lbl = 'word', countries=['US','BR','RU','KR'],                                                     words=['cow','tiger','car','dog'],                                                            sample = 1900)
X_tr4,X_te4,y_tr4,y_te4 =train_test_split(X2,Y2,test_size = 0.15)
xgb2 =XGB.XGBClassifier(max_depth=1, n_estimators=5000, learning_rate=0.25)
xgb2.fit(np.array(X_tr4),np.array(y_tr4))
score = xgb2.score(np.array(X_te4),np.array(y_te4))
print("XGBoost image recognition accuracy: {} percent".format(score*100))
# Feature importance for image recognition mode
ind2 = np.argsort(xgb2.feature_importances_)
imp2 = [X_te4.columns[i] for i in ind2]
# top 15 important features
print(imp2[-15:])


# In[45]:


X2,Y2 = data_preparer_ensemble(df_cow_en2,df_tree_en2, df_car_en2,df_mouse_en2,                                           lbl = 'word', countries=['US','BR','RU','KR'],                                                     words=['cow','tree','car','mouse'],                                                            sample = 1900)
X_tr4,X_te4,y_tr4,y_te4 =train_test_split(X2,Y2,test_size = 0.15,                                                     random_state=831713, stratify = Y2)
print(time.time())
xgb2 =XGB.XGBClassifier(max_depth=1, n_estimators=5000, learning_rate=0.25)
xgb2.fit(np.array(X_tr4),np.array(y_tr4))
print(time.time())
score = xgb2.score(np.array(X_te4),np.array(y_te4))
print(time.time())
print("XGBoost image recognition accuracy: {} percent".format(score*100))
# Feature importance for image recognition mode
ind2 = np.argsort(xgb2.feature_importances_)
imp2 = [X_te4.columns[i] for i in ind2]
# top 15 important features
print(imp2[-15:])


# In[46]:


X2,Y2 = data_preparer_ensemble(df_train_en2,df_tree_en2, df_duck_en2,df_mouse_en2,                                           lbl = 'word', countries=['US','BR','RU','KR'],                                                     words=['train','tree','duck','mouse'],                                                            sample = 1900)
X_tr4,X_te4,y_tr4,y_te4 =train_test_split(X2,Y2,test_size = 0.15,                                                     random_state=831713, stratify = Y2)
print(time.time())
xgb2 =XGB.XGBClassifier(max_depth=1, n_estimators=5000, learning_rate=0.25)
xgb2.fit(np.array(X_tr4),np.array(y_tr4))
print(time.time())
score = xgb2.score(np.array(X_te4),np.array(y_te4))
print(time.time())
print("XGBoost image recognition accuracy: {} percent".format(score*100))
# Feature importance for image recognition mode
ind2 = np.argsort(xgb2.feature_importances_)
imp2 = [X_te4.columns[i] for i in ind2]
# top 15 important features
print(imp2[-15:])


# In[48]:


X2,Y2 = data_preparer_ensemble(df_flower_en2, df_car_en2, df_school_bus_en2,df_mouse_en2,                                           lbl = 'word', countries=['US','BR','RU','KR'],                                                     words=['flower','car','school bus','mouse'],                                                            sample = 1400)
X_tr4,X_te4,y_tr4,y_te4 =train_test_split(X2,Y2,test_size = 0.15,                                                     random_state=831713, stratify = Y2)
print(time.time())
xgb2 =XGB.XGBClassifier(max_depth=1, n_estimators=5000, learning_rate=0.25)
xgb2.fit(np.array(X_tr4),np.array(y_tr4))
print(time.time())
score = xgb2.score(np.array(X_te4),np.array(y_te4))
print(time.time())
print("XGBoost image recognition accuracy: {} percent".format(score*100))
# Feature importance for image recognition mode
ind2 = np.argsort(xgb2.feature_importances_)
imp2 = [X_te4.columns[i] for i in ind2]
# top 15 important features
print(imp2[-15:])


# In[51]:


# most car
X2,Y2 = data_preparer_ensemble(df_flower_en2, df_cat_en2, df_train_en2,df_school_bus_en2,                                           lbl = 'word', countries=['US','BR','RU','KR'],                                                     words=['flower','cat','train','school bus'],                                                            sample = 1400)
X_tr4,X_te4,y_tr4,y_te4 =train_test_split(X2,Y2,test_size = 0.15,                                                     random_state=831713, stratify = Y2)
print(time.time())
xgb2 =XGB.XGBClassifier(max_depth=1, n_estimators=5000, learning_rate=0.25)
xgb2.fit(np.array(X_tr4),np.array(y_tr4))
print(time.time())
score = xgb2.score(np.array(X_te4),np.array(y_te4))
print(time.time())
print("XGBoost image recognition accuracy: {} percent".format(score*100))
# Feature importance for image recognition mode
ind2 = np.argsort(xgb2.feature_importances_)
imp2 = [X_te4.columns[i] for i in ind2]
# top 15 important features
print(imp2[-15:])


# In[52]:


X2,Y2 = data_preparer_ensemble(df_dog_en2, df_tiger_en2, df_cat_en2,df_lion_en2,                                           lbl = 'word', countries=['US','BR','RU','KR'],                                                     words=['dog','tiger','cat','lion'],                                                            sample = 1900)
X_tr4,X_te4,y_tr4,y_te4 =train_test_split(X2,Y2,test_size = 0.15,                                                     random_state=831713, stratify = Y2)
print(time.time())
xgb2 =XGB.XGBClassifier(max_depth=1, n_estimators=5000, learning_rate=0.25)
xgb2.fit(np.array(X_tr4),np.array(y_tr4))
print(time.time())
score = xgb2.score(np.array(X_te4),np.array(y_te4))
print(time.time())
print("XGBoost image recognition accuracy: {} percent".format(score*100))
# Feature importance for image recognition mode
ind2 = np.argsort(xgb2.feature_importances_)
imp2 = [X_te4.columns[i] for i in ind2]
# top 15 important features
print(imp2[-15:])


# In[53]:


X2,Y2 = data_preparer_ensemble(df_tree_en2, df_duck_en2, df_mouse_en2,df_car_en2,                                           lbl = 'word', countries=['US','BR','RU','KR'],                                                     words=['tree','duck','mouse','car'],                                                            sample = 1900)
X_tr4,X_te4,y_tr4,y_te4 =train_test_split(X2,Y2,test_size = 0.15,                                                     random_state=831713, stratify = Y2)
print(time.time())
xgb2 =XGB.XGBClassifier(max_depth=1, n_estimators=5000, learning_rate=0.25)
xgb2.fit(np.array(X_tr4),np.array(y_tr4))
print(time.time())
score = xgb2.score(np.array(X_te4),np.array(y_te4))
print(time.time())
print("XGBoost image recognition accuracy: {} percent".format(score*100))
# Feature importance for image recognition mode
ind2 = np.argsort(xgb2.feature_importances_)
imp2 = [X_te4.columns[i] for i in ind2]
# top 15 important features
print(imp2[-15:])


# In[54]:


X2,Y2 = data_preparer_ensemble(df_cat_en2, df_duck_en2, df_tree_en2,df_car_en2,                                           lbl = 'word', countries=['US','BR','RU','KR'],                                                     words=['cat','duck','tree','car'],                                                            sample = 1900)
X_tr4,X_te4,y_tr4,y_te4 =train_test_split(X2,Y2,test_size = 0.15,                                                     random_state=831713, stratify = Y2)
print(time.time())
xgb2 =XGB.XGBClassifier(max_depth=1, n_estimators=5000, learning_rate=0.25)
xgb2.fit(np.array(X_tr4),np.array(y_tr4))
print(time.time())
score = xgb2.score(np.array(X_te4),np.array(y_te4))
print(time.time())
print("XGBoost image recognition accuracy: {} percent".format(score*100))
# Feature importance for image recognition mode
ind2 = np.argsort(xgb2.feature_importances_)
imp2 = [X_te4.columns[i] for i in ind2]
# top 15 important features
print(imp2[-15:])


# In[56]:


def _country_initial_fixer(df,country,limit):
    if df[df['countrycode']==country].count()[0] > limit:
        df_c = df[df['countrycode']==country]
        random_c = np.random.choice(list(df_c.index), limit, replace=False)
        df_c = df_c.loc[list(random_c)]
    else:
        df_c = df[df['countrycode']==country]
    return df_c


# In[102]:


from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import pickle
np.random.seed(32113)

def data_preparer_ensemble(df1,df2,df3,df4, lbl = 'word', countries=['US','BR','RU','KR'],                   words=['cat','tiger','lion','dog'],sample=3000, limit = 5000):

    if lbl == 'word':
      #runs _df_initial_fixer for each word to prepare dataframe
        df_test1 = _df_initial_fixer(df1,words[0],sample)
        df_test2 = _df_initial_fixer(df2,words[1],sample)
        df_test3 = _df_initial_fixer(df3,words[2],sample)
        df_test4 = _df_initial_fixer(df4,words[3],sample)
        print(len(df_test1),len(df_test2),len(df_test3),len(df_test4))

        # convining all 4 dataframe to create a new dataframe. the new_df will be the input for XGB.
        new_df = pd.concat([df_test1,df_test2,df_test3,df_test4], axis =0)
        yd = new_df.pop('countrycode')
        Y = new_df.pop('word')
        b_loon={}
        for i in range(len(words)):
            b_loon[words[i]] = i
        # Y will be the label for my XGB model.
        Y = Y.map(b_loon)
        return new_df,Y
        
    elif lbl == 'countrycode':
      #runs _df_initial_fixer_cc for each word to prepare dataframe
        df_test1 = _df_initial_fixer_cc(df1,words[0])
        df_test2 = _df_initial_fixer_cc(df2,words[1])
        df_test3 = _df_initial_fixer_cc(df3,words[2])
        df_test4 = _df_initial_fixer_cc(df4,words[3])
        print(len(df_test1),len(df_test2),len(df_test3),len(df_test4))
        new_df = pd.concat([df_test1,df_test2,df_test3,df_test4], axis =0)
        
        #filter dataframe by selected countries
        df_cf = new_df[(new_df['countrycode']==countries[0])|(new_df['countrycode']==countries[1])|                   (new_df['countrycode']==countries[2])|(new_df['countrycode']==countries[3])]
        print(len(df_cf))

        # US
        df_country_1 = _country_initial_fixer(df_cf,countries[0],limit)
        #BR
        df_country_2 = _country_initial_fixer(df_cf,countries[1],limit)
        #RU
        df_country_3 = _country_initial_fixer(df_cf,countries[2],limit)
        #KR
        df_country_4 = _country_initial_fixer(df_cf,countries[3],limit)

        print("number of images for country {}:{}, {}:{}, {}:{}, {}:{}\n"                    .format(countries[0],len(df_country_1),countries[1],len(df_country_2),countries[2],len(df_country_3),countries[3],len(df_country_4)))
        # new_df will be the input Dataframe for XGBoost and Y will be the label
        new_df = pd.concat([df_country_1,df_country_2,df_country_3,df_country_4], axis=0)
        Y = new_df.pop('countrycode')
        b_loon = {}
        for i in range(len(countries)):
            b_loon[countries[i]] = i
        Y = Y.map(b_loon)
       
        # creating additional feature called word. In this feature number represents the word of image.
        b_loon2={'cat':0,'tiger':1,'lion':2,'dog':3,"car":5,"cow":6,"train":7,"school bus":8, "tree":9, "duck":10,"flower":11,"mouse":12,"camera":13,"computer":14,"house plant":15,"grass":16,"bush":17}
        new_df['word']=new_df['word'].map(b_loon2)

        return new_df,Y,df_country_1,df_country_2,df_country_3,df_country_4
    else:
        print("set your lbl to 'word' or 'countrycode' ")

def _df_initial_fixer(df, word, sample=3000):
    print("total number of images for df_{}: {}".format(word, len(df)))
    random_index = np.random.choice(list(df.index), sample, replace=False)
    df = df.loc[list(random_index)]
    df_test = df.drop(['drawing','key_id','timestamp','recognized','X','Y','time',                        'X_per_stroke','Y_per_stroke','time_per_stroke',                        'total_time_of_stroke','dp_per_stroke','dp_percent_per_stroke',                        'direction'], axis=1)
    return df_test

def _df_initial_fixer_cc(df, word):
    df_test = df.drop(['drawing','key_id','timestamp','recognized','X','Y','time',                        'X_per_stroke','Y_per_stroke','time_per_stroke',                        'total_time_of_stroke','dp_per_stroke','dp_percent_per_stroke',                        'direction'], axis=1)
    return df_test


# In[97]:


df_camera_en2 = pd.read_pickle('./data/MY_feature_camera.pkl')
df_computer_en2 = pd.read_pickle('./data/MY_feature_computer.pkl')
df_house_plant_en2 = pd.read_pickle('./data/MY_feature_house plant.pkl')
df_grass_en2 = pd.read_pickle('./data/MY_feature_grass.pkl')
df_bush_en2 = pd.read_pickle('./data/MY_feature_bush.pkl')


# In[103]:


X2,Y2 = data_preparer_ensemble(df_camera_en2, df_computer_en2, df_house_plant_en2,df_grass_en2,                                           lbl = 'word', countries=['US','BR','RU','KR'],                                                     words=['camera','computer','house plant','grass'],                                                            sample = 1200)
X_tr4,X_te4,y_tr4,y_te4 =train_test_split(X2,Y2,test_size = 0.15,                                                     random_state=831713, stratify = Y2)
print(time.time())
xgb2 =XGB.XGBClassifier(max_depth=1, n_estimators=5000, learning_rate=0.25)
xgb2.fit(np.array(X_tr4),np.array(y_tr4))
print(time.time())
score = xgb2.score(np.array(X_te4),np.array(y_te4))
print(time.time())
print("XGBoost image recognition accuracy: {} percent".format(score*100))
# Feature importance for image recognition mode
ind2 = np.argsort(xgb2.feature_importances_)
imp2 = [X_te4.columns[i] for i in ind2]
# top 15 important features
print(imp2[-15:])


# In[104]:


# plant
X2,Y2 = data_preparer_ensemble(df_grass_en2, df_bush_en2, df_house_plant_en2,df_flower_en2,                                           lbl = 'word', countries=['US','BR','RU','KR'],                                                     words=['grass','bush','house plant','flower'],                                                            sample = 1220)
X_tr4,X_te4,y_tr4,y_te4 =train_test_split(X2,Y2,test_size = 0.15,                                                     random_state=831713, stratify = Y2)
print(time.time())
xgb2 =XGB.XGBClassifier(max_depth=1, n_estimators=5000, learning_rate=0.25)
xgb2.fit(np.array(X_tr4),np.array(y_tr4))
print(time.time())
score = xgb2.score(np.array(X_te4),np.array(y_te4))
print(time.time())
print("XGBoost image recognition accuracy: {} percent".format(score*100))
# Feature importance for image recognition mode
ind2 = np.argsort(xgb2.feature_importances_)
imp2 = [X_te4.columns[i] for i in ind2]
# top 15 important features
print(imp2[-15:])


# In[105]:


# two animals two plants
X2,Y2 = data_preparer_ensemble(df_grass_en2, df_bush_en2, df_cat_en2,df_dog_en2,                                           lbl = 'word', countries=['US','BR','RU','KR'],                                                     words=['grass','bush','cat','dog'],                                                            sample = 1900)
X_tr4,X_te4,y_tr4,y_te4 =train_test_split(X2,Y2,test_size = 0.15,                                                     random_state=831713, stratify = Y2)
print(time.time())
xgb2 =XGB.XGBClassifier(max_depth=1, n_estimators=5000, learning_rate=0.25)
xgb2.fit(np.array(X_tr4),np.array(y_tr4))
print(time.time())
score = xgb2.score(np.array(X_te4),np.array(y_te4))
print(time.time())
print("XGBoost image recognition accuracy: {} percent".format(score*100))
# Feature importance for image recognition mode
ind2 = np.argsort(xgb2.feature_importances_)
imp2 = [X_te4.columns[i] for i in ind2]
# top 15 important features
print(imp2[-15:])


# In[106]:


# two electronics two animals
X2,Y2 = data_preparer_ensemble(df_computer_en2, df_camera_en2, df_cat_en2,df_dog_en2,                                           lbl = 'word', countries=['US','BR','RU','KR'],                                                     words=['computer','camera','cat','dog'],                                                            sample = 1900)
X_tr4,X_te4,y_tr4,y_te4 =train_test_split(X2,Y2,test_size = 0.15,                                                     random_state=831713, stratify = Y2)
print(time.time())
xgb2 =XGB.XGBClassifier(max_depth=1, n_estimators=5000, learning_rate=0.25)
xgb2.fit(np.array(X_tr4),np.array(y_tr4))
print(time.time())
score = xgb2.score(np.array(X_te4),np.array(y_te4))
print(time.time())
print("XGBoost image recognition accuracy: {} percent".format(score*100))
# Feature importance for image recognition mode
ind2 = np.argsort(xgb2.feature_importances_)
imp2 = [X_te4.columns[i] for i in ind2]
# top 15 important features
print(imp2[-15:])


# In[ ]:


# two electronics two plants
X2,Y2 = data_preparer_ensemble(df_computer_en2, df_camera_en2, df_grass_en2,df_bush_en2,                                           lbl = 'word', countries=['US','BR','RU','KR'],                                                     words=['computer','camera','grass','bush'],                                                            sample = 2600)
X_tr4,X_te4,y_tr4,y_te4 =train_test_split(X2,Y2,test_size = 0.15,                                                     random_state=831713, stratify = Y2)
print(time.time())
xgb2 =XGB.XGBClassifier(max_depth=1, n_estimators=1000, learning_rate=0.25)
xgb2.fit(np.array(X_tr4),np.array(y_tr4))
print(time.time())
score = xgb2.score(np.array(X_te4),np.array(y_te4))
print(time.time())
print("XGBoost image recognition accuracy: {} percent".format(score*100))
# Feature importance for image recognition mode
ind2 = np.argsort(xgb2.feature_importances_)
imp2 = [X_te4.columns[i] for i in ind2]
# top 15 important features
print(imp2[-15:])


# In[ ]:


# two electronics two cars
X2,Y2 = data_preparer_ensemble(df_school_bus_en2, df_car_en2, df_grass_en2,df_bush_en2,                                           lbl = 'word', countries=['US','BR','RU','KR'],                                                     words=['school bus','car','grass','bush'],                                                            sample = 1900)
X_tr4,X_te4,y_tr4,y_te4 =train_test_split(X2,Y2,test_size = 0.15,                                                     random_state=831713, stratify = Y2)
print(time.time())
xgb2 =XGB.XGBClassifier(max_depth=1, n_estimators=5000, learning_rate=0.25)
xgb2.fit(np.array(X_tr4),np.array(y_tr4))
print(time.time())
score = xgb2.score(np.array(X_te4),np.array(y_te4))
print(time.time())
print("XGBoost image recognition accuracy: {} percent".format(score*100))
# Feature importance for image recognition mode
ind2 = np.argsort(xgb2.feature_importances_)
imp2 = [X_te4.columns[i] for i in ind2]
# top 15 important features
print(imp2[-15:])


# In[ ]:




