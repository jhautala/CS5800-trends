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