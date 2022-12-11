#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 19:48:16 2022

These models are intended to set rough bounds for financial performance.

@author: jhautala
"""

import numpy as np
from numpy.random import Generator, PCG64

# internal
from util import spy, ndaq
from util.model import Model, default_budget


# Randomly buy or sell 1 share each day
class JHRandom(Model):
    def __init__(
            self,
            budget=default_budget,
            seed=42,
    ):
        super().__init__(budget)
        self.rng = Generator(PCG64(seed))
    
    def decide(self, snapshot):
        return self.rng.choice([-1, 0, 1])

class JHRandomProp(Model):
    def __init__(
            self,
            budget=default_budget,
            prop=1,
            seed=42,
    ):
        super().__init__(budget)
        self.rng = Generator(PCG64(seed))
        self.prop = prop
    
    def decide(self, snapshot):
        decision = self.rng.choice([-1, 0, 1])
        if decision > 0:
            return self.prop * np.ceil(self.balance//snapshot[-1,0])
        elif decision < 0:
            return round(self.prop) * self.shares
        else:
            return 0

# Buy at the minimum (using all budget) and sell all at the maximum... if only we had a crystal ball
class JHOmniscientMinMax(Model):
    def decide(self, snapshot):
        price = snapshot[-1,0]
        if price == spy.min_price:
            n = self.balance//price
            return n
        elif price == spy.max_price:
            n = -self.shares
            return n
        else:
            return 0
        
class JHOmniscientMinMaxNdaq(Model):
    def decide(self, snapshot):
        price = snapshot[-1,0]
        if price == ndaq.min_price:
            n = self.balance//price
            return n
        elif price == ndaq.max_price:
            n = -self.shares
            return n
        else:
            return 0