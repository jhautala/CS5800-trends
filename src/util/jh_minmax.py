#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:51:27 2022

@author: jhautala
"""


import numpy as np
from collections import deque

# internal
from util.model import Model, default_budget

class JHMinMax(Model):
    def __init__(
            self,
            budget=default_budget,
            window=728,
            conserve=True,
    ):
        super().__init__(budget)
        self.window = window
        self.conserve = conserve
        
        self.num_bought = 0
        self.tot_cost = 0
        self.min = None
        self.max = None
        self.imin = None
        self.imax = None
        self.vals = deque(maxlen=window) if window else None
    
    def decide(self, snapshot):
        price = snapshot[-1]
        
        if self.window is not None:
            # apply window
            if len(snapshot) > self.window:
                # check to see if min or max is leaving the window
                if self.vals[0] == self.min:
                    self.min = None
                if self.vals[0] == self.max:
                    self.max = None
                if self.min is None or self.max is None:
                    for i in range(1, len(self.vals)):
                        if self.min is None or self.min > self.vals[i]:
                            self.min = self.vals[i]
                        if self.max is None or self.max < self.vals[i]:
                            self.max = self.vals[i]
            
            # append to vals, pushing vals[0] out of window
            self.vals.append(price)
        
        # min and max
        ismin = False
        ismax = False
        if self.min is None or self.min > price:
            ismin = True
            self.min = price
        if self.max is None or self.max < price:
            ismax = True
            self.max = price
        
        # TODO: try making no decision when both ismin and ismax
        #       (expect marginal impact, only for the first decision)
        if ismin:
            x = int(self.balance//price)
        elif ismax:
            x = -self.shares
        else:
            x = 0
        
        cost = price * x
        if cost > self.balance:
            x = int(self.balance//price)
            cost = price * x
        elif x < -self.shares:
            x = -self.shares
            cost = price * x
        
        # check held stock value to make sure not to sell at a loss
        if x < 0 and self.conserve and price < self.tot_cost/self.num_bought:
            x = 0
            cost = 0
        
        # update held stock value
        self.tot_cost += cost
        self.num_bought += x
        
        return x

class JHMinMax_tuned(JHMinMax):
    def __init__(
            self,
            budget=default_budget,
            window=728,
            conserve=True,
    ):
        super().__init__(
            budget,
            window=window,
            conserver=conserve,
        )