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
            scale=1,
            window=100,
            z_thresh=None,
            conserve=False,
            min_profit=None,
    ):
        super().__init__(budget)
        self.window = window
        self.z_thresh = z_thresh
        self.conserve = conserve
        self.min_profit = min_profit
        self.sum = 0
        self.sumSq = 0
        self.count = 0
        self.mu = None
        self.sd = None
        self.scale = scale
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
                self.sumSq/(self.count-1) - self.count*self.mu**2/(self.count-1)
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
        
        if x < 0:
            # check held stock value to make sure not to sell at a loss
            avg_price = self.tot_cost/self.num_bought
            perceived_value = max(avg_price, self.mu)
            if self.min_profit is not None:
                perceived_value *= self.min_profit
            if self.conserve and price < perceived_value:
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

class JHNormProb_tuned(JHNormProb):
    def __init__(
            self,
            budget=default_budget,
            min_profit=1.1,
    ):
        super().__init__(
            budget,
            # scale=1.496,
            scale=0.24323232,
            min_profit=min_profit,
        )

class JHNormThresh(JHNormProb):
    def __init__(
            self,
            budget=default_budget,
            window=100,
            pct=200/3,
            min_profit=None,
    ):
        super().__init__(
            budget=budget,
            window=window,
            z_thresh=norm.ppf(.01 * pct),
            min_profit=min_profit,
        )

class JHNormThresh_tuned(JHNormThresh):
    def __init__(
            self,
            budget=default_budget,
            window=1000,
            pct=91.75,
            min_profit=None,
    ):
        super().__init__(
            budget=budget,
            window=window,
            pct=pct,
            min_profit=min_profit,
        )
