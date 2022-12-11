#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 19:10:28 2022

@author: jhautala
"""

import pandas as pd
import numpy as np

class Trend():
    def __init__(self, name='SPY'):
        self.name = name
        
        # Extract the data
        df = pd.read_csv(
                f'data/{name}.csv',
                parse_dates = ['Date']
            ).sort_values(by='Date')
        self.df = df
        
        # Convert data into unraveled lists
        _data = []
        _vol = []
        for day in df['Date']:
            row = df[df['Date'] == day]
            _data.append(row['Open'])
            _data.append(row['Close'])
            _vol.append(np.nan)
            _vol.append(row['Volume'].to_numpy()[0])
        one_dim = np.array(_data).ravel()
        self.one_dim = one_dim
        self.two_dim = np.stack([one_dim, np.array(_vol)], axis=1)
        
        # Find global min and global max to use in OmniscientMinMax
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
        self.optimal_buy = optimal_buy
        self.optimal_sell = optimal_sell
        self.min_price = min_price
        self.max_price = max_price
        
        # ----- create a DataFrame with estimated timestamps
        _dates = []
        for date in df['Date']:
            type(date)
            # # more accurate? open/close times
            # dates.append(date + pd.Timedelta(minutes=570)) # open
            # dates.append(date + pd.Timedelta(hours=16)) # close
            
            # regular interval open/close times
            _dates.append(date + pd.Timedelta(hours=6)) # open
            _dates.append(date + pd.Timedelta(hours=18)) # close
        
        self.df_w_dates = pd.DataFrame(
            self.two_dim,
            columns=['Price', 'Volume'],
            index=_dates,
        )
