#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 00:46:00 2022

@author: jhautala
"""


import numpy as np

# internal
from util.model import Model, default_budget

class ReactiveStdDev(Model):
    def __init__(
            self,
            budget=default_budget,
            shares_per_sd=300,
            window=100,
    ):
        super().__init__(budget)
        self.shares_per_sd = shares_per_sd
        self.window = window
        self.sum = 0
        self.sumSq = 0
        self.count = 0
        self.mu = None
        self.sd = None
    
    def decide(self, snapshot):
        price = snapshot[-1]
        
        # running stats
        self.sum += price
        self.sumSq += price ** 2
        self.count += 1
        
        # apply window
        if self.window is not None and self.count > self.window:
            self.sum -= snapshot[-self.count]
            self.sumSq -= snapshot[-self.count]**2
            self.count -= 1
        
        # calculate mean
        self.mu = self.sum/self.count
        if self.count > 1:
            # calculate std dev
            self.sd = np.sqrt(
                self.sumSq/(self.count-1) - self.mu**2/(self.count**2-self.count)
            )
            
            # calculate num std devs from prior point
            sd_diff = (snapshot[-1] - snapshot[-2])/self.sd
            
            # decide
            x = -int(sd_diff * self.shares_per_sd)
        else:
            x = 0
        
        cost = price * x
        if cost > self.balance:
            x = int(self.balance//price)
            cost = price * x
        elif x < -self.shares:
            x = -self.shares
            cost = price * x
        return x
