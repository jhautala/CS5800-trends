#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 19:39:00 2022

@author: ghoward
"""

from util.model import Model

# Buy all every morning, sell all every evening
# NOTE: odd size snapshots indicate open; even size is close
class GHBuyOpenSellClose(Model):
    def decide(self, snapshot):
        price = snapshot[-1]
        if (len(snapshot) % 2) == 0:
            n = -self.shares
        else:
            n = self.balance//price
        return n

# Buy all every evening, sell all every morning
class GHBuyCloseSellOpen(Model):
    def decide(self, snapshot):
        price = snapshot[-1]
        if (len(snapshot) % 2) == 0:
            n = self.balance//price
        else:
            n = -self.shares
        return n