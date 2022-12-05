#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 19:41:07 2022

@author: jhautala
"""

import numpy as np

# internal
from util.model import Model, default_budget


# Half-day momentum strategy
class JHBandWagon(Model):
    def decide(self, snapshot):
        if len(snapshot) < 2:
            return 0
        return int(np.sign(snapshot[-1] - snapshot[-2]))

# Buy and hold (using all budget available on day 1)
class JHLongHaul(Model):
    def decide(self, snapshot):
        return int(self.balance//snapshot[-1])

class JHReverseMomentum(Model):
    def __init__(
            self,
            budget=default_budget,
            shares_per=1,
    ):
        super().__init__(budget)
        self.shares_per = shares_per
    
    def decide(self, snapshot):
        if len(snapshot) < 2:
            return 0
        return self.shares_per * int(np.sign(snapshot[-2] - snapshot[-1]))

class JHReverseMomentum_tuned(JHReverseMomentum):
    def __init__(
            self,
            budget=default_budget,
    ):
        super().__init__(budget, shares_per=18)
