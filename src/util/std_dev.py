#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 00:46:00 2022

@author: jhautala
"""


import numpy as np
from scipy.stats import norm

# internal
from util.model import Model, default_budget

class ReactiveStdDev(Model):
    def __init__(
            self,
            budget=default_budget,
            shares_per_sd=300,
            scale=10,
            window=100,
    ):
        super().__init__(budget)
        self.initial_budget = budget
        self.shares_per_sd = shares_per_sd
        self.window = window
        self.sum = 0
        self.sumSq = 0
        self.count = 0
        self.mu = None
        self.sd = None
        self.scale = scale
        
        # tmp
        self.min = None
        self.max = None
        self.mins = []
        self.maxs = []
        self.mus = []
        self.sds = []
        self.zs = []
        self.sd_diffs = []
        self.costs = []
        self.pp = []
        self.sigma_mus = []
    
    def decide(self, snapshot):
        price = snapshot[-1]
        n = len(snapshot)
        
        # min and max
        if self.min is None or self.min > price:
            self.min = price
        if self.max is None or self.max < price:
            self.max = price
        self.mins.append(self.min)
        self.maxs.append(self.max)
        
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
        self.mus.append(self.mu)
        if self.count > 1:
            # calculate std dev
            self.sd = np.sqrt(
                self.sumSq/(self.count-1) - self.mu**2/(self.count**2-self.count)
            )
            self.sds.append(self.sd)
            
            # calculate z score
            z = (price - self.mu)/self.sd
            self.zs.append(z)
            
            # calculate probability
            p = norm.cdf(z)
            self.pp.append(p)
            
            # calculate num std devs from prior point
            sd_diff = (snapshot[-1] - snapshot[-2])/self.sd
            self.sd_diffs.append(sd_diff)
            
            # incorporate uncertainty around estimated mu?
            sigma_mu = self.sd/np.sqrt(n)
            self.sigma_mus.append(sigma_mu)
            
            # decide
            if self.shares_per_sd is not None:
                x = -int(sd_diff * self.shares_per_sd)
            else:
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
            cost = self.balance if self.clip else 0
            x = int(round(cost/price))
        elif x < -self.shares:
            x = -self.shares if self.clip else 0
            cost = price * x
        self.costs.append(price * x)
        return x
