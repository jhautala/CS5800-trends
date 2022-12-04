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
            scale=300,
            window=100,
    ):
        super().__init__(budget)
        self.scale = scale
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
            
            return -int(sd_diff * self.scale)
        else:
            # not enough information to make a decision yet
            return 0
