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
            scale=1,
            window=100,
            conserve=False,
    ):
        super().__init__(budget)
        self.scale = scale
        self.window = window
        self.conserve = conserve
        
        # members to track sample stats
        self.sum = 0
        self.sumSq = 0
        self.count = 0
        self.mu = None
        self.sd = None
        
        # members to track perceived value (based on purchase price)
        self.num_bought = 0
        self.tot_cost = 0
        self.avg_price = None
    
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
            
            x = -int(sd_diff * self.scale)
        else:
            # not enough information to make a decision yet
            x = 0
        
        # check intent vs budget
        # NOTE: This is redundant to Model.evaluate,
        #       but we need it for the 'conserve' feature.
        cost = price * x
        if cost > self.balance:
            x = int(self.balance//price)
            cost = price * x
        elif x < -self.shares:
            x = -self.shares
            cost = price * x
        
        # check held stock value to make sure not to sell at a loss
        if x < 0 and self.conserve and price < self.avg_price:
            x = 0
            cost = 0
        
        # update held stock value
        self.tot_cost += cost
        self.num_bought += x
        self.avg_price = self.tot_cost/self.num_bought\
            if self.num_bought\
            else None
        
        return x
