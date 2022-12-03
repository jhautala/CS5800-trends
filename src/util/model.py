#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 19:08:12 2022

TODO: Delete the 'clip' argument? I don't think there's any use
        for it in its current form...

@author: jhautala
"""

default_budget = 1e4

class Model:
    '''
    Do: subclass and override the 'decide' method.
    Do not: modify 'budget', 'balance' nor 'shares' directly
    '''
    def __init__(
            self,
            budget=default_budget,
            clip=True,
    ):
        self.balance = self.budget = budget     # Balance == budget at the start
        self.shares = 0                         # Start with no shares
        self.clip = clip                        # False to noop on impossible transactions
        self.equity = 0
        self.volume = 0
        self.volume_shares = 0
        self.trades = []
        self.net_values = []
    
    def decide(self, snapshot):
        '''
        Parameters
        ----------
        snapshot : available history
            DESCRIPTION.

        Returns
        -------
        int
            How many shares to buy (negative to sell).

        '''
        return 0
    
    def evaluate(self, snapshot):
        # record current total value
        self.net_values.append(self.get_net_value())
        
        x = self.decide(snapshot)                       # Share count to transact
        price = snapshot[-1]                            # Current price is at the end of the list
        cost = x * price                                
        # print(
        #     f'intend to {"buy" if x >= 0 else "sell"} {abs(x)} shares '
        #     f'{"+" if x <= 0 else "-"}{abs(cost):.2f}'
        # )
        if cost > self.balance:                         # Scenarios where desired purchase is too costly
            # print(f'you can\'t afford that (cost={cost}; balance={self.balance})')
            if self.clip:                               # Buy only what balance allows
                x = int(round(self.balance/price))
            else:                                       # Cancel transaction entirely
                x = 0
            cost = x * price
        elif x < -self.shares:                          # Scenarios where desired sale is more than shares owned
            # print(f'you don\'t have enough shares (x={x}; balance={self.shares})')
            if self.clip:                               # Sell only what is owned
                x = self.shares
            else:                                       # Cancel transaction entirely
                x = 0
            cost = x * price
        self.shares += x                                # Update count of shares owned
        self.balance -= cost                            # Update balance
        self.equity = self.shares * price               # Update valuation of equity
        self.volume += abs(cost)                        # Update total volume traded
        self.volume_shares += abs(x)                    # Update total count of shares traded
        self.trades.append(x)
        return x
    
    def get_net_value(self):
        return self.balance - self.budget + self.equity # Reducing by budget lets us reflect net change
                                                        # Note this means the worst case scenario is -budget
