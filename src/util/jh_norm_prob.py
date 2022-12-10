#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 20:59:36 2022

@author: jhautala
"""


import numpy as np
from scipy.stats import norm

# internal
from util.model import Model, default_budget

class JHNormProb(Model):
    def __init__(
            self,
            budget=default_budget,
            scale=10,
            window=100,
    ):
        super().__init__(budget)
        self.window = window
        self.sum = 0
        self.sumSq = 0
        self.count = 0
        self.mu = None
        self.sd = None
        self.scale = scale
    
    def decide(self, snapshot):
        price = snapshot[-1,0]
        
        # running stats
        self.sum += price
        self.sumSq += price ** 2
        self.count += 1
        
        # apply window
        if self.window is not None and self.count > self.window:
            self.sum -= snapshot[-self.count,0]
            self.sumSq -= snapshot[-self.count,0]**2
            self.count -= 1
        
        # calculate mean
        self.mu = self.sum/self.count
        if self.count > 1:
            # calculate std dev
            self.sd = np.sqrt(
                self.sumSq/(self.count-1) - self.mu**2/(self.count**2-self.count)
            )
            
            # calculate z score
            z = (price - self.mu)/self.sd
            
            # calculate probability
            p = norm.cdf(z)
            
            # decide
            cen = 2 * (p - .5)
            if cen < 0:
                # NOTE: using p here like this is more
                #       like tanh, but we might want something
                #       more like sinh...
                ceil = self.balance/price
                x = -int(self.scale * ceil * cen)
            else:
                floor = self.shares
                x = -int(self.scale * floor * cen)
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

class JHNormProb_tuned(JHNormProb):
    def __init__(
            self,
            budget=default_budget,
    ):
        super().__init__(budget, scale=1.496)
