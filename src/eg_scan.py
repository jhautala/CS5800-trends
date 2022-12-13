# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# internal
from util import spy, ndaq
from util.co_vol_and_price import COVolumeAndPrice
from util.jh_norm_prob import JHNormProb
from util.jh_std_dev import JHReactiveStdDev
from util.jh_std_dev_detail import JHStdDevDetail
from util.paramscan import scan

results = scan(spy, JHReactiveStdDev, { 'scale': np.linspace(100, 450, 1000)})
results = scan(ndaq, JHReactiveStdDev, { 'scale': np.linspace(0, 200, 1000)})

# scan(
#     spy,
#     JHStdDevDetail,
#     {
#         'mode': ['sd_diff'],
#         'scale': np.linspace(.1, 500,10),
#     },
# )


pricevol_params = {
    'buy_low': [True, False],
    'buy_slow': [True, False],
}
for trend in [spy, ndaq]:
    results = scan(trend, COVolumeAndPrice, pricevol_params)
    print('All results:')
    print(results, '\n')
