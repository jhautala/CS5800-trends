#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 19:39:06 2022

@author: jhautala
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# internal
from util import spy, ndaq
from util.jh_norm_prob import JHNormProb, JHNormThresh
from util.jh_std_dev import JHReactiveStdDev
from util.jh_std_dev_detail import JHStdDevDetail
from util.jh_simple import JHReverseMomentum
from util.paramscan import scan


alt_mid = {
    k: 0 for k in [
        'decision shares',
        'decision costs',
        'z-scores',
        'num SDs from prior',
    ]
}
alt_mid['norm probs'] = .5
incl_detail = set(['decision shares', 'decision costs'])
alt_int = set(['decision shares'])

def plot(
        trend,
        desc,
        model,
        mid=None,
        alt='decision shares',
        save_fig=False,
):  
    if alt == 'decision shares':
        second = model.trades[1:]
    elif alt == 'decision costs':
        second = model.costs[1:]
    elif alt == 'z-scores':
        second = model.zs # looks like a good input to decisions...
    elif alt == 'norm probs':
        second = norm.cdf(model.zs)
    elif alt == 'std devs':
        second = model.sds
    elif alt == 'sigma_mus':
        second = model.sigma_mus
    elif alt == 'num SDs from prior':
        second = model.sd_diffs
    elif alt == 'net value':
        second = model.net_values[1:]
    elif alt == 'overs':
        second = model.overs[1:]
    elif alt == 'overshares':
        second = model.overshares[1:]
    else:
        raise Exception(f'Invalid alt keyword "{alt}"')
    
    
    
    xx = range(len(trend.one_dim)-1)
    prices = trend.one_dim[1:]
    price_color = 'tab:blue'
    alt_color = 'tab:orange'
    
    if alt in incl_detail:
        fig, [ax1, ax3] = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(12, 8),
            sharex=True,
        )
    else:
        fig, ax1 = plt.subplots(
            figsize=(12, 6),
            sharex=True,
        )
    ax1.set_xlabel('Day')
    
    # plot mid +/- 1 sd
    if mid is not None:
        sds = np.array(model.sds)
        if mid == 'mu':
            mid = np.array(model.mus)[1:]
        else:
            mid = trend.one_dim[1:]
        ax1.fill_between(
            xx,
            mid + sds,
            mid - sds,
            alpha=0.2,
            color=price_color
        )
        ax1.plot(xx, mid + sds, '--', color=price_color, label='$\pm$1 std dev')
        ax1.plot(xx, mid - sds, '--', color=price_color)
    
    # plot price
    ax1.plot(
        xx,
        prices,
        c=price_color,
        alpha=.5,
    )
    
    # plot min/max
    # ax1.axvline(maxx, linestyle="--", color=".5", label='best time to sell')
    # ax1.axhline(model.max, linestyle="--", color=".5")
    # ax1.axvline(minx, linestyle="--", color=".5", label='best time to buy')
    # ax1.axhline(model.min, linestyle="--", color=".5")
    
    ax1.set_xlabel('time')
    ax1.set_ylabel('price')
    ax1.spines['left'].set_color(price_color)
    ax1.yaxis.label.set_color(price_color)
    ax1.tick_params(
        axis='y',
        which='both',
        color=price_color,
        labelcolor=price_color,
    )
    
    # plot secondary axis
    ax2 = ax1.twinx()
    if alt in alt_mid:
        ax2.axhline(alt_mid.get(alt), linestyle="--", color=".5")
    ax2.plot(
        xx,
        second,
        color=alt_color,
        alpha=.5
    )
    ax2.set_ylabel(alt)
    ax2.spines['right'].set_color(alt_color)
    ax2.yaxis.label.set_color(alt_color)
    ax2.tick_params(
        axis='y',
        which='both',
        color=alt_color,
        labelcolor=alt_color,
    )
    if alt in alt_int:
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # add title
    model_params = [
        f'budget={model.budget}',
        f'window={model.window}',
    ]
    if model.scale is not None:
        model_params.append(f'scale={model.scale}')
    if hasattr(model, 'conserve'):
        model_params.append(f'conserve={model.conserve}')
    
    net_perf = model.get_net_value()
    net_perf = f'{"-" if net_perf < 0 else ""}${abs(net_perf):.2f}'
    title = [
        f'{desc} price vs {alt} on {trend.name}',
        f'params: {"; ".join(model_params)}',
    ]
    if alt in incl_detail or alt == 'net value':
        title.extend([
            f'Net Fincancial Performance: {net_perf}',
        ])
        
    plt.title(' \n '.join(title))
    
    # ----- plot net value
    if alt in incl_detail:
        ax3.plot(
            xx,
            model.net_values[1:],
            color='black',
        )
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Net Value')
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(
            f'figs/{trend.name}_var_{desc}-price_vs_{alt}.png',
            dpi=300,
            bbox_inches='tight'
        )
    plt.show()

def run_model(trend, model, desc=None, save_fig=False):
    for i in range(1, trend.two_dim.shape[0]+1):
        model.evaluate(trend.two_dim[:i,:].copy())
    print(f'financial performance: {model.get_net_value()}')

    full_desc = f'{type(model).__name__}'
    if desc is not None:
        full_desc += f'_{desc}'
    # plot(trend, full_desc, model, alt='sigma_mus', save_fig=save_fig)
    plot(trend, full_desc, model, alt='std devs', save_fig=save_fig)
    plot(trend, full_desc, model, alt='num SDs from prior', save_fig=save_fig)
    plot(trend, full_desc, model, alt='z-scores', save_fig=save_fig)
    # plot(trend, full_desc, model, alt='norm probs', save_fig=save_fig)
    plot(trend, full_desc, model, alt='net value', save_fig=save_fig)
    plot(trend, full_desc, model, alt='overs', save_fig=save_fig)
    # plot(trend, full_desc, model, alt='overshares', save_fig=save_fig)
    plot(trend, full_desc, model, save_fig=save_fig)
    plot(trend, full_desc, model, alt='decision costs', save_fig=save_fig)

def main():
    save_fig = False
    trend = ndaq
    trend = spy
    
    model = JHReactiveStdDev()
    run_model(trend, model, save_fig=save_fig)
    
    model = JHStdDevDetail(scale=68.6, conserve=True)
    run_model(trend, 'sd_diffs_conserve', model, save_fig=save_fig)
    
    model = JHStdDevDetail(mode='normprob')
    run_model(trend, 'norm', model, save_fig=save_fig)
    
    model = JHStdDevDetail(mode='normprob', scale=1.496)
    run_model(trend, 'norm_cheat', model, save_fig=save_fig)
    
    model = JHStdDevDetail(mode='normprob', scale='max')
    run_model(trend, 'norm_minmax', model, save_fig=save_fig)
    
    # ----- try minmax with 1-year window
    model = JHStdDevDetail(mode='minmax', window=728)
    run_model(trend, 'minmax', model, save_fig=save_fig)
    
    model = JHStdDevDetail(mode='minmax', window=728, conserve=True)
    run_model(trend, 'minmax', model, save_fig=save_fig)
    
    
    # model = JHStdDevDetail(
    #     scale=158.33333333,
    # )
    # for i in range(1, trend.two_dim.shape[0]+1):
    #     model.evaluate(trend.two_dim[:i,:].copy())
    # net_value = model.get_net_value()
    
    # for i in range(1, trend.two_dim.shape[0]+1):
    #     model.evaluate(trend.two_dim[:i,:].copy())
    # plot('sd_diffs', model, alt='sigma_mus', save_fig=save_fig)
    
    
    # fig, ax1 = plt.subplots(
    #     figsize=(12, 6),
    #     sharex=True,
    # )
    
    # ax1.plot(
    #     range(len(model.costs)),
    #     model.costs,
    #     color='tab:blue',
    #     alpha=.5
    # )
    # ax1.plot(
    #     range(len(model.costs)),
    #     model.overs,
    #     color='tab:orange',
    #     alpha=.5
    # )
    # plt.show()
    
    
    # # ----- find best params for JHNormProb
    # results = []
    # for scale in np.linspace(0,2,2000):
    #     model = JHNormProb(
    #         scale=scale,
    #     )
    #     for i in range(1, trend.two_dim.shape[0]+1):
    #         model.evaluate(trend.two_dim[:i,:].copy())
    #     value = model.get_net_value()
    #     results.append([scale, value])
    # results = np.array(results)
    # argmax = np.argmax(results[:,1])
    # print(f'{argmax}: {results[argmax,:]}')
    
    # plt.scatter(x=results[:,0], y=results[:,1])
    # plt.title(f'{type(model)}')
    # plt.show()
    
    # ----- new way
    # results = scan(spy, JHNormProb, { 'scale': np.linspace(.24, .25, 100)})
    
    # # ----- find best params for JHNormThresh
    # results = []
    # for pct in np.linspace(90,92,100):
    #     model = JHNormThresh(
    #         pct=pct,
    #     )
    #     for i in range(1, trend.two_dim.shape[0]+1):
    #         model.evaluate(trend.two_dim[:i,:].copy())
    #     value = model.get_net_value()
    #     results.append([pct, value])
    # results = np.array(results)
    # argmax = np.argmax(results[:,1])
    # print(f'{argmax}: {results[argmax,:]}')
    
    # plt.scatter(x=results[:,0], y=results[:,1])
    # plt.show()
    
    
    # gs_results = []
    # for window in range(1000, 1001):
    #     for pct in np.linspace(52,67):
    #         model = JHNormThresh(
    #             window=window,
    #             pct=pct,
    #         )
    #         for i in range(1, trend.two_dim.shape[0]+1):
    #             model.evaluate(trend.two_dim[:i,:].copy())
    #         value = model.get_net_value()
    #         gs_results.append([window, pct, value])
    # gs_results = np.array(gs_results)
    
    # tmp = gs_results[:,2]>10000
    # # tmp = gs_results[:,0]==600
    # tmp = gs_results[tmp]
    # df2 = pd.DataFrame(tmp, columns=['window', 'pct', 'net_value'])

    # sns.scatterplot(df2, x='window', y='pct', hue='net_value')
    # plt.show()
    
    # argmax = np.argmax(gs_results[:,2])
    # print(f'{argmax}: {gs_results[argmax,:]}')
    # # 49816: [ 1001.            55.55102041 11388.120186  ]
    # # 16: [ 1100.            55.55102041 11096.900629  ]
    
    # plt.scatter(x=gs_results[:,0], y=gs_results[:,2])
    # plt.show()
    
    
    # trend = ndaq
    # trend = spy
    # model = JHReactiveStdDev(scale=1.68468468)
    # for i in range(1, trend.two_dim.shape[0]+1):
    #     model.evaluate(trend.two_dim[:i,:].copy())
    # model.get_net_value()
    
    # # ----- find best params for JHReactiveStdDev
    # results = []
    # for scale in np.linspace(.1, 10,1000):
    #     model = JHReactiveStdDev(
    #         scale=scale,
    #     )
    #     for i in range(1, trend.two_dim.shape[0]+1):
    #         model.evaluate(trend.two_dim[:i,:].copy())
    #     value = model.get_net_value()
    #     results.append([scale, value])
    # results = np.array(results)
    # argmax = np.argmax(results[:,1])
    # print(f'{argmax}: {results[argmax,:]}')
    
    # plt.scatter(x=results[:,0], y=results[:,1])
    # plt.show()
    
    
    # # ----- find best params for ReactiveGreedy
    # results = []
    # for shares_per in range(1, 5000):
    #     model = ReverseMomentum(
    #         shares_per=shares_per,
    #     )
    #     for i in range(1, trend.two_dim.shape[0]+1):
    #         model.evaluate(trend.two_dim[:i,:].copy())
    #     value = model.get_net_value()
    #     results.append([shares_per, value])
    # results = np.array(results)
    # argmax = np.argmax(results[:,1])
    # print(f'{argmax}: {results[argmax,:]}')
    
    # plt.scatter(x=results[:,0], y=results[:,1])
    # plt.show()
    
    
    # # ----- find best params for JHStdDevDetail
    # mode = 'sd_diff'
    # mode = 'normprob'
    
    # results = []
    # for scale in np.linspace(
    #         1,
    #         50
    #         # 68.535,
    #         # 68.65,
    #         # 68:70 -> 14: [  68.57142857 4337.069705  ]
    #         # 60:100 -> 11: [  68.97959184 4259.179675  ]
    #         # .1:500 -> 8: [  81.71632653 3727.479573  ]
    #         # ----- normprob
    #         # 1.49,
    #         # 1.5,
    #         # 1:5 -> 6: [1.48979592e+00 2.69312972e+03]
    #         # 2:3 -> 6: [2.12244898e+00 2.69259995e+03]
    #         # .1:50 -> 2: [2.13673469e+00 2.66500996e+03]
    #         # .1:100 -> 1: [2.13877551e+00 2.66357996e+03]
    #         # .1:200 -> 1: [   4.17959184 2506.299616  ]
    # ):
    #     model = JHStdDevDetail(
    #         mode=mode,
    #         scale=scale,
    #     )
    #     for i in range(1, trend.two_dim.shape[0]+1):
    #         model.evaluate(trend.two_dim[:i,:].copy())
    #     value = model.get_net_value()
    #     results.append([scale, value])
    # results = np.array(results)
    # argmax = np.argmax(results[:,1])
    # print(f'{argmax}: {results[argmax,:]}')
    
    # plt.scatter(x=results[:,0], y=results[:,1])
    # plt.show()
    
    # for i in range(results.shape[0]):
    #     if np.isclose(results[i,1], 2.69312972e+03):
    #         print(f'{i}: {results[i,:]}')
    
    # for i in range(results.shape[0]):
    #     if np.isclose(results[i,1], 4502.859698):
    #         print(f'{i}: {results[i,0]}')
    # results[32,:]
        
    #     tmp = pd.DataFrame(results)
    #     tmp.describe()
    
    # def test_best():
    #     scales = [
    #         5.40816327e+15,
    #         # 5.57038734e+15,
    #         # 5.82653061e+15,
    #         # 2089983722656843.0
    #     ]
    #     for i, scale in enumerate(scales):
    #         model = JHStdDevDetail(mode='normprob', scale=scale)
    #         for i in range(1, trend.two_dim.shape[0]+1):
    #             model.evaluate(trend.two_dim[:i,:].copy())
    #         print(f'financial performance: {model.get_net_value()}')
        
    #         run_model(f'norm_{i}', model, save_fig=save_fig)

if __name__ == "__main__":
    main()