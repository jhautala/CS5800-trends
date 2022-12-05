#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 19:48:16 2022

These models are intended to set rough bounds for financial performance.

@author: jhautala
"""

from numpy.random import Generator, PCG64

# internal
from util.model import Model, default_budget
from util.data import min_price, max_price


# Randomly buy or sell 1 share each day
class JHRandom(Model):
    def __init__(
            self,
            budget=default_budget,
            seed=42,
    ):
        super().__init__(budget)
        self.rng = Generator(PCG64(seed))
    
    def decide(self, snapshot):
        return self.rng.choice([-1, 1])

# Buy at the minimum (using all budget) and sell all at the maximum... if only we had a crystal ball
class JHOmniscientMinMax(Model):
    def decide(self, snapshot):
        price = snapshot[-1]
        if price == min_price:
            n = self.balance//price
            return n
        elif price == max_price:
            n = -self.shares
            return n
        else:
            return 0