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
        if len(snapshot) < 10:
            return 0
        return int(np.sign(snapshot[-10] - snapshot[-1]))
