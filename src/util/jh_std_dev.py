#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 00:46:00 2022

@author: jhautala
"""


import numpy as np

# internal
from util.model import Model, default_budget

class JHReactiveStdDev(Model):
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
                (self.sumSq - self.count*self.mu**2)/(self.count-1)
            )
            
            # calculate num std devs from prior point
            sd_diff = (snapshot[-1,0] - snapshot[-2,0])/self.sd
            spend = -sd_diff * self.scale * self.balance
            x = int(spend//price)
            
            # NOTE: this is what I actually meant to do, but
            #       I accidentally found a better model, by
            #       using an asymmetric buy/sell policy...
            # prop = -sd_diff * self.scale
            # if sd_diff > 0:
            #     x = int(prop * self.shares)
            # else:
            #     spend = prop * self.balance
            #     x = int(spend//price)
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
        
        if x < 0:
            # check held stock value to make sure not to sell at a loss
            avg_price = self.tot_cost/self.num_bought
            if self.conserve and price < avg_price:
                x = 0
                cost = 0
            
            # update value of held stock
            self.tot_cost += x * avg_price
        elif x > 0:
            # update value of held stock
            self.tot_cost += cost
        
        # update held stock count
        self.num_bought += x
        
        return x

class JHReactiveStdDev_tuned(JHReactiveStdDev):
    def __init__(
            self,
            budget=default_budget,
            window=100,
    ):
        super().__init__(
            budget,
            scale=158.33333333,
            window=window,
        )

class JHReactiveStdDev_ndaq(JHReactiveStdDev):
    def __init__(
            self,
            budget=default_budget,
            window=100,
    ):
        super().__init__(
            budget,
            scale=76.24624624624624,
            window=window,
        )
