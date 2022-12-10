

"""
Created on Sat Dec 10

@author: mmacavoy
"""

from util.model import Model

class MMbuytrendpos(model):

    # if the current price is above the price 2 days before buy all at open and sell all at close

    def decide(self, snapshot):
        price = snapshot[-1]
        if(snapshot[-1] >= snapshot[-3]):
            n = self.balance//price
        else:
            n  = -self.shares
        return n
