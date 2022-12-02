#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 23:39:39 2022

@author: jhautala
"""

import argparse
import timeit
import tracemalloc
import numpy as np
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt
plt.ioff() # disable interactive plotting

# internal
from util.model import Model, default_budget
from util.data import one_dim
from util.std_dev import ReactiveStdDev


# ----- arg parsing
parser = argparse.ArgumentParser(
    prog = 'CS5800 trends - evaluate models',
    description = 'Compare different trading models',
    epilog = 'Text at the bottom of help',
)
parser.add_argument(
    '--time-performance-iterations',
    metavar='',
    type=int,
    default=0,
    help='iterations per model for measuring time performance'
)
parser.add_argument(
    '--include-plots',
    type=bool,
    default=False,
    help='option to display plots of prices vs decisions'
)
parser.add_argument(
    '--save-figs',
    type=bool,
    default=False,
    help='option to save plots of prices vs decisions'
)

# - extract args
args=parser.parse_args()
time_perf_iter = args.time_performance_iterations
include_plots = args.include_plots
save_figs = args.save_figs


# ----- models
# Randomly buy or sell 1 share each day
class Random(Model):
    def __init__(
            self,
            budget=default_budget,
            seed=42, # Meaning of life
    ):
        super().__init__(budget)
        self.rng = Generator(PCG64(seed))
    
    def decide(self, snapshot):
        return int(np.round(np.clip(self.rng.standard_normal(), -1, 1)))

# Buy 1 share every day until budget has been exceeded
class OptimisticGreedy(Model):
    '''
    This model just wants to buy one share each day.
    '''
    def decide(self, snapshot):
        return 1

# Buy and hold (using all budget available on day 1)
class LongHaul(Model):
    def decide(self, snapshot):
        if len(snapshot) == 1:
            return int(self.budget//snapshot[0])
        else:
            return 0

# Half-day momentum strategy
class BandWagon(Model):
    def decide(self, snapshot):
        if len(snapshot) < 2:
            return 0
        return int(np.sign(snapshot[-1] - snapshot[-2]))

# Half-day reverse momentum strategy
class ReactiveGreedy(Model):
    def decide(self, snapshot):
        if len(snapshot) < 2:
            return 0
        return int(np.sign(snapshot[-2] - snapshot[-1]))

# Buy a the minimum (using all budget) and sell all at the maximum... if only we had a crystal ball
class OmniscientMinMax(Model):
    def decide(self, snapshot):
        price = snapshot[-1]
        if price == 214.767181:
            n = self.budget//price
            return n
        elif price == 479.220001:
            n = -self.shares
            return n
        else:
            return 0

# One week reverse momentum strategy
class BuyTheDip(Model):
    def decide(self, snapshot):
        if len(snapshot) < 10:
            return 0
        return int(np.sign(snapshot[-10] - snapshot[-1]))

# Buy all every morning, sell all every evening
class BuyOpenSellClose(Model):
    def decide(self, snapshot):
        price = snapshot[-1]
        if (len(snapshot) % 2) == 0:
            n = self.budget//price
        else:
            n = -self.shares
        return n

# Buy all every evening, sell all every morning
class BuyCloseSellOpen(Model):
    def decide(self, snapshot):
        price = snapshot[-1]
        if (len(snapshot) % 2) == 0:
            n = -self.shares
        else:
            n = self.budget//price
        return n

# these models were hand-tuned to optimize the rate of buying/selling
class ReactiveGreedy_cheat(Model):
    def decide(self, snapshot):
        if len(snapshot) < 2:
            return 0
        return 42 * int(np.sign(snapshot[-2] - snapshot[-1]))

class ReactiveStdDev_cheat(ReactiveStdDev):
    def __init__(
            self,
            budget=default_budget,
    ):
        super().__init__(budget, shares_per_sd=485, window=100)

def evaluate_model(
        data,
        model_type,
        budget=default_budget,
        skip_perf=False,
):
    n = len(data)
    model = model_type(budget)
    # TODO: use local vars for verification?
    # start = model.balance
    # curr = start
    # shares = 0
    for i in range(1, n+1):
        model.evaluate(data[:i].copy())
    return model

def plot_decisions(
        model,
        time_perf_ms=None,
        show_plot=False,
        save_fig=False,
):
    xx = range(len(one_dim))
    
    price_color = 'tab:blue'
    trade_color = 'tab:orange'

    fig, ax1 = plt.subplots(
        figsize=(12, 6),
        sharex=True,
    )
    ax1.set_xlabel('Day')
    
    # plot price
    ax1.plot(
        xx,
        one_dim,
        c=price_color,
        alpha=.5,
    )
    ax1.set_ylabel('Price')
    ax1.yaxis.label.set_color(price_color)
    ax1.spines['left'].set_color(price_color)
    ax1.tick_params(
        axis='y',
        which='both',
        color=price_color,
        labelcolor=price_color,
    )
    
    # plot decisions
    ax2 = ax1.twinx()
    ax2.axhline(0, linestyle="--", color=".5")
    ax2.plot(
        xx,
        model.trades,
        color=trade_color,
        alpha=.5,
        linestyle='',
        marker='.',
    )
    ax2.set_ylabel('Decisions')
    ax2.spines['right'].set_color(trade_color)
    ax2.yaxis.label.set_color(trade_color)
    ax2.tick_params(
        axis='y',
        which='both',
        color=trade_color,
        labelcolor=trade_color,
    )
    
    # add title
    model_name = type(model).__name__
    net_perf = model.get_value() - model.budget
    net_perf = f'{"-" if net_perf < 0 else ""}${abs(net_perf):.2f}'
    title = [
        f'{model_name} Price vs Decisions',
        f'Net Fincancial Performance: {net_perf}',
    ]
    if time_perf_ms is not None:
        title.append(f'Time Performance: {time_perf_ms:.3f} ms')
    plt.title(' \n '.join(title))
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(
            f'figs/price_vs_decisions_{model_name}.png',
            dpi=300,
            bbox_inches='tight'
        )
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

# ----- main execution
def main():
    # # TODO delete these argument override
    # include_plots = True
    # save_figs = True
    
    results = []
    for model_type in [
            BuyOpenSellClose,
            BuyCloseSellOpen,
            Random,
            OptimisticGreedy,
            BandWagon,
            ReactiveGreedy,
            LongHaul,
            OmniscientMinMax,
            BuyTheDip,
            ReactiveStdDev,
            ReactiveGreedy_cheat,
            ReactiveStdDev_cheat,
    ]:
        model_name = model_type.__name__
        # print(f'trying {model_name}')
        
        # timeit*1000 to convert time to milliseconds
        time_perf_ms = None\
            if time_perf_iter == 0\
            else timeit.timeit(
                    lambda: evaluate_model(one_dim, model_type),
                    number=time_perf_iter,
                )*1000/time_perf_iter
        model = evaluate_model(one_dim, model_type)
        results.append([
            model_name,
            model,
            model.get_value(),
            time_perf_ms,
        ])
        if include_plots:
            plot_decisions(
                model,
                time_perf_ms=time_perf_ms,
                show_plot=True,
                save_fig=save_figs,
            )
    
    results = np.array(results)
    
    print('financial performance:')
    for i in np.flip(np.argsort(results[:,2])):
        [model_name, model, score, time_perf_ms] = results[i]
        print(
            f'\t{model_name}:\n\t\t'
            f'{model.balance:.2f} - {model.budget:.2f} + {model.equity:.2f} = '
            f'{"-" if score < 0 else ""}${abs(score):.2f}'
        )
    
    if time_perf_iter > 0:
        print(
            f'time performance {time_perf_iter} '
            f'iteration{"" if time_perf_iter == 1 else "s"}:'
        )
        for i in np.argsort(results[:,3]):
            [model_name, model, score, time_perf_ms] = results[i]
            print(
                f'\t{model_name} performance:\n\t\t'
                f'{time_perf_ms:.3} ms'
            )

if __name__ == "__main__":
    main()