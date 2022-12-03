#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 19:39:06 2022

@author: jhautala
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# internal
from util.data import one_dim
from util.std_dev_detail import StdDevDetail


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

def plot(
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
    
    
    
    xx = range(len(one_dim)-1)
    prices = one_dim[1:]
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
            mid = one_dim[1:]
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
    
    # add title
    model_params = [
        f'budget={model.budget}',
        f'window={model.window}',
    ]
    if model.shares_per_sd is not None:
        model_params.append(f'shares_per_sd={model.shares_per_sd}')
    else:
        model_params.append(f'scale={model.scale}')
    model_name = type(model).__name__
    net_perf = model.get_net_value()
    net_perf = f'{"-" if net_perf < 0 else ""}${abs(net_perf):.2f}'
    title = [
        f'{model_name} price vs {alt}',
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
            f'figs/std_dev-{desc}-price_vs_{alt}.png',
            dpi=300,
            bbox_inches='tight'
        )
    plt.show()

def run_model(desc, model, save_fig=False):
    for i in range(1, len(one_dim)+1):
        model.evaluate(one_dim[:i].copy())
    print(f'financial performance: {model.get_net_value()}')
    
    # tmp0 = one_dim.copy()[:-1]
    # tmp1 = one_dim.copy()[1:]
    # tmp2 = tmp1 - tmp0
    # tmp2.min()
    # pd.DataFrame(tmp2).describe()

    plot(desc, model, alt='std devs', save_fig=save_fig)
    plot(desc, model, alt='num SDs from prior', save_fig=save_fig)
    plot(desc, model, alt='z-scores', save_fig=save_fig)
    # plot(desc, model, alt='norm probs', save_fig=save_fig)
    plot(desc, model, alt='net value', save_fig=save_fig)
    plot(desc, model, alt='overs', save_fig=save_fig)
    # plot(desc, model, alt='overshares', save_fig=save_fig)
    plot(desc, model, save_fig=save_fig)
    plot(desc, model, alt='decision costs', save_fig=save_fig)
    # plot(desc, model, alt='sigma_mus', save_fig=save_fig)


def main():
    save_fig = False
    
    # best model...
    # tmp = BuyOpenSellClose()
    # for i in range(1, len(one_dim)+1):
    #     tmp.evaluate(one_dim[:i].copy())
    # tmp.get_net_value()
    
    # previous best...
    # model = StdDevDetail(shares_per_sd=485)
    # wtf! this does better than previous brute force calculation...
    # model = StdDevDetail(shares_per_sd=2089983722656843.0)
    
    model = StdDevDetail(shares_per_sd=600)
    run_model('sd_diffs', model, save_fig=save_fig)
    
    model = StdDevDetail(shares_per_sd=1e19)
    run_model('sd_diffs_cheat', model, save_fig=save_fig)
    
    model = StdDevDetail(shares_per_sd=None, scale=1)
    run_model('norm', model, save_fig=save_fig)
    
    model = StdDevDetail(shares_per_sd=None, scale=600)
    run_model('norm_moderate', model, save_fig=save_fig)
    
    model = StdDevDetail(shares_per_sd=None, scale=5.40816327e+15)
    run_model('norm_cheat', model, save_fig=save_fig)
        
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
    
    
    # ----- find best params for StdDevDetail with sd diffs
    # results = []
    # for shares_per_sd in np.linspace(
    #     1e18,
    #     1e19,
    #     # 1e18,
    #     # 9e18,
    #     # Out[142]: array([9.00000000e+18, 1.22097914e+21])
    #     1e18,
    #     1e19,
    #     # Out[145]: array([1.00000000e+19, 1.34313175e+21])
    # ):
    #     model = StdDevDetail(shares_per_sd=shares_per_sd)
    #     for i in range(1, len(one_dim)+1):
    #         model.evaluate(one_dim[:i].copy())
    #     value = model.get_net_value()
    #     results.append([shares_per_sd, value])
    # results = np.array(results)
    # np.argmax(results[:,1])
    # results[49,:]
    
    # ----- find best params for StdDevDetail with norm prob
    # def find():
    #     results = []
    #     for scale in np.linspace(
    #         5.2e+15,
    #         5.615,
    #     ):
    #         model = StdDevDetail(shares_per_sd=None, scale=scale)
    #         for i in range(1, len(one_dim)+1):
    #             model.evaluate(one_dim[:i].copy())
    #         value = model.get_net_value()
    #         results.append([scale, value])
    #     results = np.array(results)
    #     np.argmax(results[:,1])
        
    #     results[3,:]
        
        
    #     plt.scatter(x=results[:,0], y=results[:,1])
    #     plt.show()
        
        
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
    #         model = StdDevDetail(shares_per_sd=None, scale=scale)
    #         for i in range(1, len(one_dim)+1):
    #             model.evaluate(one_dim[:i].copy())
    #         print(f'financial performance: {model.get_net_value()}')
        
    #         run_model(f'norm_{i}', model, save_fig=save_fig)

if __name__ == "__main__":
    main()