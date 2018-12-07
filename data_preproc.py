# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 21:03:26 2018

@author: JoshJin
"""

import pandas as pd
import numpy as np
from ast import literal_eval
import warnings
warnings.filterwarnings("ignore")

def read_data(path):
    '''
    :param path: the path to the csv file to read
    :return: top 10000 entries
    '''
    df = pd.read_csv(path, skiprows=1, names=["country", "drawing", "key_id", "recognized", "timestamp", "word"])
    return df.head(10000)


def get_x(l):
    '''
    :param l: a list of [x, y, t]'s
    :return: only the x in one long list
    '''
    x = []
    for stroke in l:
        x = x + ([round(k) for k in stroke[0]])
    return x


def get_y(l):
    '''
    :param l: a list of [x, y, t]'s
    :return: only the y in one long list
    '''
    y = []
    for stroke in l:
        y = y + ([round(k) for k in stroke[1]])
    return y


def array_normalizer(array1, Xmin, Xmax, array_min):
    '''
    :param array1: array that you want to normalize (1D array or list)
    :param Xmin: minimum value of your X array (int)
    :param Xmax: maximum value of your X array (int)
    :param array_min: minimum value of array1
    :return: normalized array of array1
    '''
    return (np.array(array1)-np.array([array_min]*len(array1)))/float(Xmax-Xmin)


def process_df(df):
    '''
    :param df: a pandas dataframe in the raw format of quickdraw database
    :return: a dataframe with normalized entries
    '''
    df = df[(df['recognized']==1)]
    df['drawing'] = df['drawing'].apply(literal_eval)
    df['stroke_count'] = df['drawing'].map(lambda x: len(x))
    df['x'] = df['drawing'].apply(get_x)
    df['y'] = df['drawing'].apply(get_y)
    df = df[(df['stroke_count']<10)]
    x_new = {}
    y_new = {}
    y_max = {}
    for i in df.index:
        x = df.loc[i, 'x']
        y = df.loc[i, 'y']
        x_mintemp = np.min(x) - 10
        x_maxtemp = np.max(x) + 10
        y_mintemp = np.min(y) - 10
        x_norm = array_normalizer(x, x_mintemp, x_maxtemp, x_maxtemp)
        y_norm = array_normalizer(y, x_mintemp, x_maxtemp, y_mintemp)
        x_new[i] = x_norm
        y_new[i] = y_norm
        y_max[i] = np.max(y_norm)
    df['x_norm'] = pd.Series(x_new)
    df['y_norm'] = pd.Series(y_new)
    df['y_max'] = pd.Series(y_max)
    return df


def convert_df_into_image(df):
    '''
    :param df: dataframe from process_df
    :return: a dataframe with column ('word', 'image', 'country')
             word: the class of the image
             image: an image in np array of (1, 42*42), with value of 1 or 0
             country: country of the doodler
    '''
    df.index = range(len(df))
    images = {}
    for ind in df.index:
        image = np.zeros((42, 42))
        x_array = np.around(np.array(df.loc[ind, 'x_norm']) * 42)
        y_array = np.around(np.array(df.loc[ind, 'y_norm']) * 42 / float(df.loc[ind, 'y_max']))
        x_array[x_array>=28.] = 27
        y_array[y_array>=42.] = 41
        for i in range(len(x_array)):
            image[int(np.around(y_array[i])), int(np.around(x_array[i]))] = 1
        images[ind] = image.reshape(1, 42*42)
    df['image'] = pd.Series(images)
    result_df = df[['word', 'image', 'country']].copy()
    return result_df.dropna()


'''
Uncomment to test functions from above
'''
# car_df = read_data("519_refined_data/data/car.csv")
# car_df = process_df(car_df)
# car_df = convert_df_into_image(car_df)
# print(car_df.head(10))
