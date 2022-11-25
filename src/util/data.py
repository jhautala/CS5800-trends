#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 19:10:28 2022

@author: jhautala
"""

import pandas as pd
import numpy as np

df = pd.read_csv('data/SPY.csv')\
    .sort_values(by='Date')

_data = []
for day in df['Date']:
    row = df[df['Date'] == day]
    _data.append(row['Open'])
    _data.append(row['Adj Close'])
one_dim = np.array(_data).ravel()


optimal_buy = None
optimal_sell = None
min_price = None
max_price = None
for i, d in enumerate(one_dim):
    if min_price is None or min_price > d:
        # print(f'{i}: {d}')
        min_price = d
        optimal_buy = i
    if max_price is None or max_price < d:
        max_price = d
        optimal_sell = i
# NOTE: luckily, for this dataset the global min preceds global max
#       therefore the optimal strategy is simple
# optimal_buy = 1169
# optimal_sell = 2070