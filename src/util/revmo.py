#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 20:50:34 2022

@author: jhautala
"""

import numpy as np

# internal
from util.model import Model, default_budget

class ReactiveGreedy(Model):
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
