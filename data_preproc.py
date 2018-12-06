# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 21:03:26 2018

@author: JoshJin
"""

import pandas as pd
import numpy as np
from ast import literal_eval

def read_data(path):
    df = pd.read_csv(path, skiprows=1, names=["country", "drawing", "key_id", "recognized", "timestamp", "word"])
    return df.head(10000)


def get_x(l):
    for stroke in l:
        print(1)


def process_df(df):
    df = df[(df['recognized']==1)]
    df['drawing'] = df['drawing'].apply(literal_eval)
    df['stroke_count'] = df['drawing'].map(lambda x: len(x))
    df['x'] = df['drawing'].map(lambda x: x[:][0])
    print(df['x'].tolist()[0])
    print(df['drawing'].tolist()[0])
#    drawing_list = df['drawing'].tolist()
#    x_list = []
#    y_list = []
#    len_list = []
#    for drawing in drawing_list:
#    print(drawing_list[0][0])
# tmp
car_df = read_data("519_refined_data/data/car.csv")
process_df(car_df)