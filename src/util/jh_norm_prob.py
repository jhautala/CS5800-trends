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
            z_thresh=None,
            conserve=True,
    ):
        super().__init__(budget)
        self.window = window
        self.z_thresh = z_thresh
        self.sum = 0
        self.sumSq = 0
        self.count = 0
        self.mu = None
        self.sd = None
        self.scale = scale
        self.conserve = conserve
        self.num_bought = 0
        self.tot_cost = 0
    
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
            
            # calculate z score
            z = (price - self.mu)/self.sd
            
            # decide
            if self.z_thresh is not None:
                if z >= self.z_thresh:
                    x = -self.shares
                elif z < -self.z_thresh:
                    x = self.balance//price
                else:
                    x = 0
            else:
                # calculate probability
                p = norm.cdf(z)
                
                cen = 2 * (p - .5)
                if cen < 0:
                    # NOTE: this curve 
                    ceil = self.balance/price
                    x = -int(self.scale * ceil * cen)
                else:
                    floor = self.shares
                    x = -int(self.scale * floor * cen)
        else:
            x = 0
        
        # enforce limits, so we can keep accurate track
        # of the value of our held stock
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

class JHNormProb_tuned(JHNormProb):
    def __init__(
            self,
            budget=default_budget,
    ):
        super().__init__(budget, scale=1.496)

class JHNormThresh(JHNormProb):
    def __init__(
            self,
            budget=default_budget,
            window=100,
            pct=60,
    ):
        super().__init__(
            budget=budget,
            window=window,
            scale=1.496,
            z_thresh=norm.ppf(.01 * pct),
        )

class JHNormThresh_tuned(JHNormThresh):
    def __init__(
            self,
            budget=default_budget,
            window=1000,
            # pct=55.55102041,
            pct=55.67346939,
    ):
        super().__init__(
            budget=budget,
            window=window,
            pct=pct,
        )
