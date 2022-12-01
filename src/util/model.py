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
        self.balance = self.budget = budget     # Balance == budget at the start
        self.shares = 0                         # Start with no shares
        self.clip = clip                        # Clip is the 
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
            How many shares to buy (negative values to sell).

        '''
        return 0
    
    def evaluate(self, snapshot):
        n = self.decide(snapshot)                       # Share count to transact
        # print(f'intend to {"buy" if n >= 0 else "sell"} {abs(n)} shares')
        price = snapshot[-1]                            # Current price is at the end of the list
        cost = n * price                                
        if cost > self.balance:                         # Scenarios where desired purchase is too costly
            # print(f'you can\'t afford that (cost={cost}; balance={self.balance})')
            if self.clip:                               # Buy only what balance allows
                n = int(np.round(self.balance/price))
                cost = n * price
            else:                                       # Cancel transaction entirely
                return 0
        elif n < -self.shares:                          # Scenarios where desired sale is more than shares owned
            # print(f'you don\'t have enough shares (n={n}; balance={self.shares})')
            if self.clip:                               # Sell only what is owned
                n = self.shares
                cost = n * price
            else:                                       # Cancel transaction entirely
                return 0
        self.shares += n                                # Update count of shares owned
        self.balance -= cost                            # Update balance
        self.equity = self.shares * price               # Update valuation of equity
        self.volume += abs(cost)                        # Update total volume traded
        self.volume_shares += abs(n)                    # Update total count of shares traded
        return n
    
    def get_value(self):
        return self.balance - self.budget + self.equity # Reducing by budget lets us reflect net change
                                                        # Note this means the worst case scenario is -budget