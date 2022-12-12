#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One week reverse momentum strategy

@author: ghoward
"""

import numpy as np
from util.model import Model

class GHBuyTheDip(Model):
    def decide(self, snapshot):
        if snapshot.shape[0] < 10:
            return 0
        return np.sign(snapshot[-10,0] - snapshot[-1,0])
