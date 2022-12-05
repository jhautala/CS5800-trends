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
    ):
        self.balance = self.budget = budget     # Balance == budget at the start
        self.shares = 0                         # Start with no shares
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
        
        # Share count to transact
        x = int(self.decide(snapshot))

        # Current price is at the end of the list
        price = snapshot[-1,0]
        cost = x * price
        
        # make sure we have enough cash/shares for the transaction
        # print(
        #     f'intend to {"buy" if x >= 0 else "sell"} {abs(x)} shares '
        #     f'{"+" if x <= 0 else "-"}{abs(cost):.2f}'
        # )
        # Scenarios where desired purchase is too costly
        if cost > self.balance:
            # print(f'you can\'t afford that (cost={cost}; balance={self.balance})')
            x = int(self.balance//price)
            cost = x * price
        # Scenarios where desired sale is more than shares owned
        elif x < -self.shares:
            # print(f'you don\'t have enough shares (x={x}; balance={self.shares})')
            # Sell only what is owned
            x = -self.shares
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
