#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 00:46:00 2022

@author: jhautala
"""


import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_rgba

# internal
from util.data import one_dim
from util.model import Model, default_budget

class ReactiveGreedy(Model):
    def __init__(
            self,
            budget=default_budget,
            shares_per=1,
    ):
        super().__init__(budget)
        self.shares_per = shares_per
    
    def decide(self, snapshot):
        if len(snapshot) < 2:
            return 0
        return self.shares_per * int(np.sign(snapshot[-2] - snapshot[-1]))


class ReactiveStdDev(Model):
    def __init__(
            self,
            budget=default_budget,
            shares_per_sd=300,
            scale=10,
            window=100,
    ):
        super().__init__(budget)
        self.initial_budget = budget
        self.shares_per_sd = shares_per_sd
        self.window = window
        self.sum = 0
        self.sumSq = 0
        self.count = 0
        self.mu = None
        self.sd = None
        self.scale = scale
        
        # tmp
        self.min = None
        self.max = None
        self.mins = []
        self.maxs = []
        self.mus = []
        self.sds = []
        self.zs = []
        self.sd_diffs = []
        self.costs = []
        self.pp = []
        self.sigma_mus = []
    
    def decide(self, snapshot):
        price = snapshot[-1]
        n = len(snapshot)
        
        # min and max
        if self.min is None or self.min > price:
            self.min = price
        if self.max is None or self.max < price:
            self.max = price
        self.mins.append(self.min)
        self.maxs.append(self.max)
        
        # running stats
        self.sum += price
        self.sumSq += price ** 2
        self.count += 1
        
        # apply window
        if self.window is not None and self.count > self.window:
            self.sum -= snapshot[-self.count]
            self.sumSq -= snapshot[-self.count]**2
            self.count -= 1
        
        # calculate mean
        self.mu = self.sum/self.count
        self.mus.append(self.mu)
        if self.count > 1:
            # calculate std dev
            self.sd = np.sqrt(
                self.sumSq/(self.count-1) - self.mu**2/(self.count**2-self.count)
            )
            self.sds.append(self.sd)
            
            # calculate z score
            z = (price - self.mu)/self.sd
            self.zs.append(z)
            
            # calculate probability
            p = norm.cdf(z)
            self.pp.append(p)
            
            # calculate num std devs from prior point
            sd_diff = (snapshot[-1] - snapshot[-2])/self.sd
            self.sd_diffs.append(sd_diff)
            
            # incorporate uncertainty around estimated mu?
            sigma_mu = self.sd/np.sqrt(n)
            self.sigma_mus.append(sigma_mu)
            
            # decide
            if self.shares_per_sd is not None:
                x = -int(sd_diff * self.shares_per_sd)
            else:
                cen = 2 * (p - .5)
                if cen < 0:
                    # NOTE: using p here like this is more
                    #       like tanh, but we might want something
                    #       more like sinh...
                    ceil = self.balance/price
                    x = -int(self.scale * ceil * cen)
                else:
                    floor = self.shares
                    x = -int(self.scale * floor * cen)
        else:
            x = 0
        
        cost = price * x
        if cost > self.balance:
            cost = self.balance if self.clip else 0
        elif x < -self.shares:
            x = -self.shares if self.clip else 0
        self.costs.append(price * x)
        return x

class AgressiveEarly(Model):
    def __init__(
            self,
            budget=default_budget,
            shares_per_sd=300,
    ):
        super().__init__(budget)
        self.initial_budget = budget
        self.shares_per_sd = shares_per_sd
        self.sum = 0
        self.sumSq = 0
        self.mu = None
        self.sd = None
    
    def decide(self, snapshot):
        price = snapshot[-1]
        n = len(snapshot)
        
        # min and max
        if self.min is None or self.min > price:
            self.min = price
        if self.max is None or self.max < price:
            self.max = price
        self.mins.append(self.min)
        self.maxs.append(self.max)
        
        # running stats
        self.sum += price
        self.sumSq += price ** 2
        
        # calculate mean
        self.mu = self.sum/n
        if n > 1:
            # calculate std dev
            self.sd = np.sqrt(
                self.sumSq/(n-1) - self.mu**2/(n**2-n)
            )
            
            # calculate z score
            self.zs.append((price - self.mu)/self.sd)
            sd_diff = (snapshot[-1] - snapshot[-2])/self.sd
            return -int(sd_diff * self.shares_per_sd)
        else:
            return 0

alt_mid = {
    k: 0 for k in ['decision shares', 'decision costs', 'z-scores']
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
        second = np.array(model.trades)[1:]
    elif alt == 'decision costs':
        second = np.array(model.costs)[1:]
    elif alt == 'z-scores':
        second = np.array(model.zs) # looks like a good input to decisions...
    elif alt == 'norm probs':
        second = norm.cdf(np.array(model.zs))
    elif alt == 'std devs':
        second = np.array(model.sds)
    elif alt == 'sigma_mus':
        second = np.array(model.sigma_mus)
    else:
        raise Exception(f'Invalid alt keyword "{alt}"')
    
    
    
    xx = range(len(one_dim)-1)
    prices = one_dim[1:]
    price_color = 'tab:blue'
    alt_color = 'tab:orange'
    
    
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
    
    ax1.set_xlabel('day')
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
    if hasattr(alt_mid, alt):
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
    net_perf = model.get_value() - model.budget
    net_perf = f'{"-" if net_perf < 0 else ""}${abs(net_perf):.2f}'
    title = [
        f'{model_name} price vs {alt}',
    ]
    if alt in incl_detail:
        title.extend([
            f'params: {"; ".join(model_params)}',
            f'Net Fincancial Performance: {net_perf}',
        ])
        
    plt.title(' \n '.join(title))
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(
            f'figs/std_dev_{desc}_price_vs_{alt}.png',
            dpi=300,
            bbox_inches='tight'
        )
    plt.show()

def run_model(desc, model, save_fig=False):
    for i in range(1, len(one_dim)+1):
        model.evaluate(one_dim[:i].copy())
    print(f'financial performance: {model.get_value()}')

    plot(desc, model, alt='std devs', save_fig=save_fig)
    plot(desc, model, alt='z-scores', save_fig=save_fig)
    # plot(desc, model, alt='norm probs', save_fig=save_fig)
    plot(desc, model, save_fig=save_fig)
    plot(desc, model, alt='decision costs', save_fig=save_fig)
    # plot(desc, model, alt='sigma_mus', save_fig=save_fig)
    

def main():
    save_fig = True
    run_model('cheat', ReactiveStdDev(shares_per_sd=485), save_fig=save_fig)
    run_model('norm', ReactiveStdDev(shares_per_sd=None), save_fig=save_fig)
    
    
    # one_dim[0]
    # p = 10000*z_scores.mean()
    # 485/p

if __name__ == "__main__":
    main()