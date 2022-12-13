#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 00:46:00 2022

TODO: incorporate Welford's method:
    https://jonisalonen.com/2014/efficient-and-accurate-rolling-standard-deviation/
    https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/

@author: jhautala
"""


import numpy as np
from scipy.stats import norm
from collections import deque

# internal
from util.model import Model, default_budget

class JHStdDevDetail(Model):
    def __init__(
            self,
            budget=default_budget,
            mode='sd_diff',
            scale=1,
            window=100,
            conserve=False,
    ):
        super().__init__(budget)
        self.window = window
        self.mode = mode
        self.scale = scale
        self.conserve = conserve
        
        # state for tracking sample stats
        self.sum = 0
        self.sumSq = 0
        self.count = 0
        self.mu = None
        self.sd = None
        self.num_bought = 0
        self.tot_cost = 0
        
        # series data for introspection
        self.min = None
        self.max = None
        self.vals = deque(maxlen=window) if window else None
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
        self.held = []
        self.can_buy = []
    
    def decide(self, snapshot):
        price = snapshot[-1,0]
        n = snapshot.shape[0]
        
        # running stats
        self.sum += price
        self.sumSq += price ** 2
        self.count += 1
        
        if self.window is not None:
            # apply window
            if self.count > self.window:
                self.sum -= snapshot[-self.count,0]
                self.sumSq -= snapshot[-self.count,0]**2
                self.count -= 1
                
                # check to see if min or max is leaving the window
                leaving = snapshot.shape[0] - self.window - 1
                if snapshot[leaving,0] == self.min:
                    self.min = None
                if snapshot[leaving,0] == self.max:
                    self.max = None
                if self.min is None or self.max is None:
                    for i in range(leaving+1, snapshot.shape[0]):
                        if self.min is None or self.min > snapshot[i,0]:
                            self.min = snapshot[i,0]
                        if self.max is None or self.max < snapshot[i,0]:
                            self.max = snapshot[i,0]
            
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
        self.mins.append(self.min)
        self.maxs.append(self.max)
        
        # calculate mean
        self.mu = self.sum/self.count
        self.mus.append(self.mu)
        if self.count > 1:
            self.held.append(self.shares)
            self.can_buy.append(self.balance//price)
            
            # calculate std dev
            self.sd = np.sqrt(
                (self.sumSq - self.count*self.mu**2)/(self.count-1)
            )
            self.sds.append(self.sd)
            
            # calculate z score
            z = (price - self.mu)/self.sd
            self.zs.append(z)
            
            # calculate probability
            p = norm.cdf(z)
            self.pp.append(p)
            
            # calculate num std devs from prior point
            sd_diff = (snapshot[-1,0] - snapshot[-2,0])/self.sd
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
                    spend = sd_diff * self.scale * self.balance
                    x = -int(spend//price)
            elif self.mode == 'normprob': # prob
                # TODO: try different distributions (other than normal)?
                #       change shape of curve to be flatter near zero
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
            elif self.mode == 'minmax':
                # TODO: try no decision when both ismin and ismax
                if ismin:
                    x = int(self.balance//price)
                elif ismax:
                    x = -self.shares
                else:
                    x = 0
            else:
                raise Exception(f'undefined mode: "{self.mode}"')
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
        self.costs.append(price * x)
        
        return x
