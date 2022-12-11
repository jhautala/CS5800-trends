# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# internal
from util import spy
from util.jh_norm_prob import JHNormProb
from util.paramscan import scan

results = scan(spy, JHNormProb, { 'scale': np.linspace(.01, 2, 100)})