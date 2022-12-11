#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:51:27 2022

@author: jhautala
"""


import numpy as np

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
    
    def decide(self, snapshot):
        price = snapshot[-1]
        
        if self.window is not None:
            # apply window
            if len(snapshot) > self.window:
                # check to see if min or max is leaving the window
                leaving = len(snapshot) - self.window - 1
                if snapshot[leaving] == self.min:
                    self.min = None
                if snapshot[leaving] == self.max:
                    self.max = None
                if self.min is None or self.max is None:
                    for i in range(leaving+1, len(snapshot)):
                        if self.min is None or self.min > snapshot[i]:
                            self.min = snapshot[i]
                        if self.max is None or self.max < snapshot[i]:
                            self.max = snapshot[i]
        
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