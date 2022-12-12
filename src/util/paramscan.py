#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:48:42 2022

@author: jhautala
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product


def scan(trend, model_type, grid):
    '''
    Perform a scan over given params to find best financial performance.
    NOTE: This will also display a scatter plot across all parameter values
        if only one paramater name is in the grid argument.

    Parameters
    ----------
    trend : Trend
        A model of input data.
    model_type : Model
        A decision model.
    grid : dict of lists
        A dictionary of parameter names to lists of parameter values.

    Returns
    -------
    DataFrame
        A DataFrame containing all the results across all combinations of parameter values.

    '''
    items = sorted(grid.items())
    keys, values = zip(*items)
    results = []
    for vv in product(*values):
        params = dict(zip(keys, vv))
        model = model_type(**params)
        for i in range(1, trend.two_dim.shape[0]+1):
            model.evaluate(trend.two_dim[:i,:].copy())
        net_value = model.get_net_value()
        result = [net_value]
        for k,v in params.items():
            result.append(v)
        results.append(result)
    results = np.array(results)
    argmax = np.argmax(results[:,0])
    print(f'{argmax}: {", ".join(params.keys())}={results[argmax,1:]} -> {results[argmax,0]}')
    
    if len(keys) == 1:
        plt.scatter(x=results[:,1], y=results[:,0])
        plt.title(f'{type(model).__name__} per {keys[0]}')
        plt.xlabel(f'{keys[0]}')
        plt.ylabel('Net Value (USD)')
        plt.show()
    
    return pd.DataFrame(results, columns=['net_value', *keys])
