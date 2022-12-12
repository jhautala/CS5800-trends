#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 20:58:22 2022

@author: cokara
"""

import numpy as np

# internal
from util.model import Model, default_budget


# Half-day momentum strategy
class COLowAndSlow(Model):
    def __init__(
            self,
            budget=default_budget,
    ):
        super().__init__(budget)
    
    def decide(self, snapshot):
        # make sure we have enough data
        if snapshot.shape[0] < 4:
            return 0
        
        # make sure this is a close time
        curr_open = snapshot.shape[0] % 2
        if curr_open:
            return 0
        
        [currPrice, currVolume] = snapshot[-1,:]
        [prevPrice, prevVolume] = snapshot[-3,:]
        priceDir = np.sign(currPrice - prevPrice)
        volumeDir = np.sign(currVolume - prevVolume)
        if priceDir > 0 and volumeDir > 0:
            return -self.shares
        elif priceDir < 0 and volumeDir < 0:
            return self.balance//currPrice
        else:
            return 0;
