# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# internal
from util import spy
from util.jh_norm_prob import JHNormProb
from util.jh_std_dev import JHReactiveStdDev
from util.paramscan import scan

results = scan(spy, JHReactiveStdDev, { 'scale': np.linspace(1, 40, 100)})