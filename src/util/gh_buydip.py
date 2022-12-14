#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One week reverse momentum strategy

@author: ghoward
"""

import numpy as np
from util.model import Model, default_budget

class GHBuyTheDip(Model):
    def decide(self, snapshot):
        if snapshot.shape[0] < 10:
            return 0
        return np.sign(snapshot[-10,0] - snapshot[-1,0])


class GHBuyTheDip2(Model):
    def __init__(
            self,
            budget=default_budget,
            shares_per=None,
            prop=1,
    ):
        super().__init__(budget)
        self.shares_per = shares_per
        self.prop = prop
    
    def decide(self, snapshot):
        if snapshot.shape[0] < 10:
            return 0
        price = snapshot[-1,0]
        direction = np.sign(price - snapshot[-10,0])
        
        # check for 'shares_per' mode
        if self.shares_per is not None:
            return -direction * self.shares_per
        
        # apply proportional to held
        if direction > 0:
            return -self.shares * self.prop
        elif direction < 0:
            return (self.balance*self.prop)//price
        else:
            return 0
