#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 18:10:07 2022

@author: jhautala
"""

import pandas as pd

from util.data import perf_filename


# ----- config/constants
model_names = [
        'RefLongHaul',
        'RefRandomProp',
        'RefOmniscientMinMax',
        'GHBuyTheDip',
        'COLowAndSlow',
        'MMbuytrendneg',
        'JHReactiveStdDev',
        'JHReactiveStdDev_tuned',
        'JHMinMax',
]
usernames = [
        'greg',
        'jhautala',
        'maxwellmacavoy',
        'Collins',
]
dataset = 'SPY'


# ----- main execution
result_df = pd.read_csv(perf_filename)
spy_df = result_df[result_df['dataset'] == dataset]
print('\t' + '\t'.join([*usernames, 'average']))
for mn in model_names:
    row = [mn]
    mm = spy_df['model'] == mn
    for un in usernames:
        um = spy_df['username'] == un
        r = spy_df[mm & um]['time_performance'].values
        if len(r):
            row.append(f'{r[0]:.2f}')
        else:
            row.append('')
    avg = spy_df[mm]['time_performance'].dropna().mean()
    row.append(f'{avg:.2f}')
    print('\t'.join(row))