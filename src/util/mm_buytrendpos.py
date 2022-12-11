

"""
Created on Sat Dec 10

@author: mmacavoy
"""

from util.model import Model, default_budget

class MMbuytrendpos(Model):
    def __init__(
            self,
            budget=default_budget,
    ):
        super().__init__(budget)
        self.bought = False

    # if the current price is above the price 2 days before buy all at open and sell all at close
    def decide(self, snapshot):
        # wait until we have enough history to decide
        if len(snapshot) < 5:
            return 0
        
        # check to see if we bought this morning
        if self.bought:
            self.bought = False
            return -self.shares
        
        # check to see if this is open (i.e. odd-size snapshot)
        if len(snapshot) % 2:
            price = snapshot[-1]
            # check current price > today - 2 days (4 indices)
            if(snapshot[-1] > snapshot[-5]):
                self.bought = True
                return self.balance//price
        
        # either not morning or current <= 2 days prior
        return 0
