#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 19:08:12 2022

@author: jhautala
"""

import numpy as np

default_budget = 1e4

class Model:
    '''
    Do: subclass and override the 'decide' method.
    Do not: modify 'budget', 'balance' nor 'shares' directly
    '''
    def __init__(self, budget=default_budget, clip=True):
        self.balance = self.budget = budget
        self.shares = 0
        self.clip = clip
        self.equity = 0
        self.volume = 0
        self.volume_shares = 0
    
    def decide(self, snapshot):
        '''
        Parameters
        ----------
        snapshot : available history
            DESCRIPTION.

        Returns
        -------
        int
            How many shares to buy (negaive values to sell).

        '''
        return 0
    
    def evaluate(self, snapshot):
        n = self.decide(snapshot)
        # print(f'intend to {"buy" if n >= 0 else "sell"} {abs(n)} shares')
        price = snapshot[-1]
        cost = n * price
        if cost > self.balance:
            # print(f'you can\'t afford that (cost={cost}; balance={self.balance})')
            if self.clip:
                n = int(np.round(self.balance/price))
                cost = n * price
            else:
                return 0
        elif n < -self.shares:
            # print(f'you don\'t have enough shares (n={n}; balance={self.shares})')
            if self.clip:
                n = self.shares
                cost = n * price
            else:
                return 0
        self.shares += n
        self.balance -= cost
        self.equity = self.shares * price
        self.volume += abs(cost)
        self.volume_shares += abs(n)
        return n
    
    def get_value(self):
        return self.balance - self.budget + self.equity