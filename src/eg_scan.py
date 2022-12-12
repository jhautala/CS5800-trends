# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# internal
from util import spy
from util.jh_norm_prob import JHNormProb
from util.jh_std_dev import JHReactiveStdDev
from util.jh_std_dev_detail import JHStdDevDetail
from util.paramscan import scan

results = scan(spy, JHReactiveStdDev, { 'scale': np.linspace(158, 158.5, 100)})

scan(
    spy,
    JHStdDevDetail,
    {
        'mode': ['sd_diff'],
        'scale': np.linspace(.1, 500,10),
    },
)

