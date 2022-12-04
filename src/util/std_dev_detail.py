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

class StdDevDetail(Model):
    def __init__(
            self,
            budget=default_budget,
            mode='sd_diff',
            scale=1,
            window=100,
    ):
        super().__init__(budget)
        self.window = window
        self.mode = mode
        self.scale = scale
        
        # state for tracking sample stats
        self.sum = 0
        self.sumSq = 0
        self.count = 0
        self.mu = None
        self.sd = None
        
        # series data for introspection
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
        self.overs = []
        self.overshares = []
    
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
            sigma_mu = self.mu/np.sqrt(n)
            self.sigma_mus.append(sigma_mu)
            
            # decide
            if self.mode == 'sd_diff':
                if self.scale == 'max':
                    if sd_diff == 0:
                        x = 0
                    elif sd_diff > 0:
                        x = self.shares
                    else:
                        x = self.balance/price
                else:
                    x = -int(sd_diff * self.scale)
            else: # prob
                cen = 2 * (p - .5)
                if cen < 0:
                    # NOTE: using p here like this is more
                    #       like tanh, but we might want something
                    #       more like sinh...
                    ceil = self.balance/price
                    if self.scale == 'max':
                        x = ceil
                    else:
                        x = -int(self.scale * ceil * cen)
                else:
                    floor = self.shares
                    if self.scale == 'max':
                        x = floor
                    else:
                        x = -int(self.scale * floor * cen)
        else:
            x = 0
        
        cost = price * x
        if cost > self.balance:
            over = cost - self.balance
            self.overs.append(over)
            self.overshares.append(int(np.ceil(over/price)))
            x = int(self.balance//price)
            cost = price * x
        elif x < -self.shares:
            overshare = x + self.shares
            self.overshares.append(overshare)
            self.overs.append(overshare * price)
            x = -self.shares
            cost = price * x
        else:
            self.overs.append(0)
            self.overshares.append(0)
        self.costs.append(price * x)
        return x
