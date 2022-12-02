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
            window=100,
            use_mu=False, # False to use diff between curr and prev
    ):
        super().__init__(budget)
        self.initial_budget = budget
        self.shares_per_sd = shares_per_sd
        self.window = window
        self.use_mu
        self.sum = 0
        self.sumSq = 0
        self.count = 0
        self.mu = None
        self.sd = None
        
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
        self.xx = []
        self.pp = []
    
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
            
            # decide
            if self.shares_per_sd is not None:
                x = -int(sd_diff * self.shares_per_sd)
            else:
                # incorporate uncertainty around mu?
                # sigma_mu = self.sd/np.sqrt(n)
                cen = 2 * (p - .5)
                if cen < 0:
                    # NOTE: using p here like this is more
                    #       like tanh, but we might want something
                    #       more like sinh...
                    ceil = self.balance/price
                    x = -int(cen * ceil)
                else:
                    floor = self.shares
                    x = int(cen * floor)
        else:
            x = 0
        
        self.xx.append(x)
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


def main():
    # ----- find best params for StdDevReactive
    results = []
    shares_per_sds = np.linspace(5,500)
    windows = np.arange(2,500)
    params = []
    for shares_per_sd in shares_per_sds:
        for window in windows:
            model = ReactiveStdDev(
                shares_per_sd=shares_per_sd,
                window=window,
            )
            for i in range(1, len(one_dim)+1):
                model.evaluate(one_dim[:i].copy())
            
            params.append({
                'window': window,
                'shares_per_sd': shares_per_sd,
            })
            results.append([
                window,
                shares_per_sd,
                model.get_value(),
            ])
    r_df = pd.DataFrame(
        results,
        columns=['window', 'shares', 'profit'],
    )
    
    # tmp = ReactiveStdDev(
    #     shares_per_sd=5,
    #     window=100,
    # )
    # for i in range(1, len(one_dim)+1):
    #     tmp.evaluate(one_dim[:i].copy())
    
    # dir(tmp)
    # tmp.get_value()
    # vars(tmp)
    # attrs = set(dir(ReactiveStdDev)) - set(dir(tmp))
    # [a for a in attrs if not a.startswith('__') and not callable(getattr(obj, a))]
    # pp = []
    # for p in params:
    #     kk = list(p.keys())
    #     pp.append({
    #         'shares_per_sd': p.get(kk[0]),
    #         'window': int(p.get(kk[0])) if len(kk) == 1 else p.get(kk[1]),
    #     })
    
    r_df['window_per_share'] = r_df['window']/r_df['shares']
    # r_df\
    #     .drop(columns=['window_per_share'])\
    #     .to_csv('std_dev_params.csv', index=False)
    
    sns.pairplot(r_df)
    sns.scatterplot(
        r_df[r_df['window'] < 50],
        x='window',
        y='shares',
        hue='profit',
    )
    10000./one_dim[0]
    ss = sorted(r_df['shares'].unique())
    for shares in ss[30:40]:
        ax = sns.jointplot(
            r_df[r_df['shares'] == shares],
            # r_df,
            x='window',
            y='profit',
            # hue='window',
            # kind='hist',
            # alpha='.2'
        )
        plt.title(f'shares: {shares:.2f}')
        plt.tight_layout()
        plt.show()
    ww = sorted(r_df['window'].unique())
    for window in ww[10:20]:
        ax = sns.jointplot(
            r_df[r_df['window'] == window],
            # r_df,
            x='shares',
            y='profit',
            # hue='window',
            # kind='hist',
            # alpha='.2'
        )
        plt.title(f'window: {window:.2f}')
        plt.tight_layout()
        plt.show()
    
    # # min: [15, 489.8979591836735, -9909.054751000001]
    # # max: [5, 500.0, 377137.9390860005]
    # results[24405]
    # params[23917]
    # tmp = np.array(results)[:,2]
    # np.argmax(tmp)
    # tmp[23917]
    
    # # ----- scatter with alpha
    # def scatter(x, y, color, alphas, **kwarg):
    #     r, g, b, _ = to_rgba(color)
    #     color = [(r, g, b, alpha) for alpha in alphas]
    #     plt.scatter(x, y, c=color, **kwarg)
    
    # # results = np.array(results)
    # # x = results[:,0]
    # # y = results[:,1]
    # # alphas = results[:,2]
    # r_df = pd.read_csv('std_dev_params.csv')
    # x = r_df['window']
    # y = r_df['shares']
    # span = r_df['profit'].max() - r_df['profit'].min()
    # alphas = (r_df['profit'] - r_df['profit'].min())/span
    # scatter(x, y, 'tab:blue', alphas, s=10)

    # ----- find best params for GreedyReactive
    # for i in range(1000):
    #     # model = ReactiveGreedy(shares_per=i)
    #     # model = ReactiveStdDev(shares_per_sd=1. * i)
    #     for i in range(1, len(one_dim)+1):
    #         model.evaluate(one_dim[:i].copy())
    #     value = model.get_value()
    #     results.append(value)
    
    results = np.array(results)
    np.argmax(results)
    # fig, ax1 = plt.subplots()
    # ax1.plot(
    #     list(range(1000)),
    #     results,
    # )
    
    
    # model = ReactiveStdDev(shares_per_sd=485) # works well with window 100
    model = ReactiveStdDev(shares_per_sd=485)
    for i in range(1, len(one_dim)+1):
        model.evaluate(one_dim[:i].copy())
    print(f'financial performance: {model.get_value()}')
    if not hasattr(model, 'sds'):
        return
    
    minx = None
    maxx = None
    for i, d in enumerate(one_dim):
        if model.min == d:
            minx = i
        if model.max == d:
            maxx = i
    
    mid = np.array(model.mus)[1:]
    sds = np.array(model.sds)
    z_scores = np.array(model.zs)
    sd_diffs = np.array(model.sd_diffs)
    pp = norm.cdf(z_scores)
    
    one_dim[0]
    p = 10000*z_scores.mean()
    485/p
    
    
    yy = one_dim[1:]
    xx = range(len(one_dim)-1)
    train_color = 'tab:orange'
    test_color = 'tab:blue'
    
    fig, ax1 = plt.subplots()
    
    # # +/- 1 sd
    # ax1.fill_between(
    #     xx,
    #     mid + sds,
    #     mid - sds,
    #     alpha=0.2,
    #     color=test_color
    # )
    # ax1.plot(xx, mid + sds, '--', color=test_color, label='$\pm$1 std dev')
    # ax1.plot(xx, mid - sds, '--', color=test_color)
    
    # # price
    ax1.plot(
        xx,
        yy,
        c = 'tab:blue',
        alpha=.5,
    )
    
    # # min/max
    # ax1.axvline(maxx, linestyle="--", color=".5", label='best time to sell')
    # ax1.axhline(model.max, linestyle="--", color=".5")
    # ax1.axvline(minx, linestyle="--", color=".5", label='best time to buy')
    # ax1.axhline(model.min, linestyle="--", color=".5")
    
    ax1.set_ylabel('Price')
    ax1.set_xlabel('Day')
    # ax1.legend()
    ax1 = ax1.twinx()
    ax1.axhline(0, linestyle="--", color=".5")
    ax1.plot(
        xx,
        z_scores,
        color=train_color,
        alpha=.5
    )
    ax1.set_ylabel('Smoothed')
    # ax2.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()