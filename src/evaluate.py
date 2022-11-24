#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:42:36 2022

@author: jhautala
"""

import timeit
import tracemalloc
import numpy as np
from numpy.random import Generator, PCG64

# internal
from model import Model, default_budget
from data import one_dim

# ----- models
class Random(Model):
    def __init__(self, budget=default_budget, seed=42):
        super().__init__(budget)
        self.rng = Generator(PCG64(seed))
    
    def decide(self, snapshot):
        return int(np.round(np.clip(self.rng.standard_normal(), -1, 1)))

class OptimisticGreedy(Model):
    '''
    This model just wants to buy one share each day.
    '''
    def decide(self, snapshot):
        return 1

class Bandwagon(Model):
    def decide(self, snapshot):
        if len(snapshot) < 2:
            return 0
        return int(np.sign(snapshot[-1] - snapshot[-2]))

class ReactiveGreedy(Model):
    def decide(self, snapshot):
        if len(snapshot) < 2:
            return 0
        return int(np.sign(snapshot[-2] - snapshot[-1]))


# ----- the testing framework
def evaluate_model(
        data,
        model_type,
        budget=default_budget,
        skip_perf=False,
):
    model = model_type(budget)
    # # extra local vars for verification
    # start = model.balance
    # curr = start
    # shares = 0
    for i in range(1, len(data)):
        model.evaluate(data[:i].copy())
    return model

for model_type in [
        Random,
        OptimisticGreedy,
        ReactiveGreedy,
        Bandwagon,
]:
    # timeit.timeit(
    #     lambda: evaluate_model(data, model_type),
    #     number=1000
    # )
    
    model = evaluate_model(one_dim, model_type)
    print(
        f'{model_type.__name__} performance:\n\t'
        f'{model.balance} - {model.budget} + {model.equity} = {model.get_value()}'
    )