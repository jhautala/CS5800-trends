#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 23:39:39 2022

@author: jhautala
"""

import numpy as np
from numpy.random import Generator, PCG64
import timeit
import tracemalloc
import argparse

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
# parser.add_argument(
#     '--skip-time-perf',
#     metavar='',
#     type=bool,
#     default=False,
#     help='option to skip time performance measurement'
# )

# - extract args
args=parser.parse_args()
# skip_time_perf = args.skip_time_perf
time_perf_iter = args.time_performance_iterations


# ----- models
# Randomly buy or sell 1 share each day
class Random(Model):
    def __init__(self, budget=default_budget, seed=42): # Meaning of life
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
    model = model_type(budget, run_length=n)
    # # extra local vars for verification
    # start = model.balance
    # curr = start
    # shares = 0
    for i in range(1, n+1):
        model.evaluate(data[:i].copy())
    return model


# ----- main execution
def main():
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
        # convert time to milliseconds
        time_perf_ms = None\
            if time_perf_iter == 0\
            else timeit.timeit(
                    lambda: evaluate_model(one_dim, model_type),
                    number=time_perf_iter,
                )*1000/time_perf_iter
        model = evaluate_model(one_dim, model_type)
        results.append([
            model_type.__name__,
            model,
            model.get_value(),
            time_perf_ms,
        ])
    
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