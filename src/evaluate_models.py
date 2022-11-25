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

# - args
args=parser.parse_args()
# skip_time_perf = args.skip_time_perf
time_perf_iter = args.time_performance_iterations


# ----- models
class Random(Model):
    def __init__(self, budget=default_budget, seed=42):
        super().__init__(budget)
        self.rng = Generator(PCG64(seed))
    
    def decide(self, snapshot):
        return int(np.round(np.clip(self.rng.standard_normal(), -1, 1)))

class OptimisticGreedy(Model):
    '''
    This model just wants to buy one share each day.
    '''
    def decide(self, snapshot):
        return 1

class LongHaul(Model):
    def decide(self, snapshot):
        if len(snapshot) == 1:
            return int(self.budget//snapshot[0])
        else:
            return 0

class BandWagon(Model):
    def decide(self, snapshot):
        if len(snapshot) < 2:
            return 0
        return int(np.sign(snapshot[-1] - snapshot[-2]))

class ReactiveGreedy(Model):
    def decide(self, snapshot):
        if len(snapshot) < 2:
            return 0
        return int(np.sign(snapshot[-2] - snapshot[-1]))

class MinMax(Model):
    def decide(self, snapshot):
        price = snapshot[-1]
        if price == 214.767181:
            n = self.budget//price
            print(f'buying {n} at {price}')
            return n
        elif price == 479.220001:
            n = -self.shares
            print(f'selling {-n} shares at {price}')
            return n
        else:
            return 0

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


def evaluate_model(
        data,
        model_type,
        budget=default_budget,
        skip_perf=False,
):
    model = model_type(budget)
    # # extra local vars for verification
    # start = model.balance
    # curr = start
    # shares = 0
    for i in range(1, len(data)+1):
        model.evaluate(data[:i].copy())
    return model


# ----- main execution
def main():
    results = []
    for model_type in [
            Random,
            OptimisticGreedy,
            BandWagon,
            ReactiveGreedy,
            LongHaul,
            OmniscientMinMax,
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
            f'{model.balance} - {model.budget} + {model.equity} = '
            f'{"-" if score < 0 else ""}${abs(score):2f}'
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