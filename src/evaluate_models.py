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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.ioff() # disable interactive plotting

# internal
from util.model import default_budget
from util.data import one_dim, df_w_dates

# models
# TODO: find a way to simplify all these imports? seems like a lot
from util.gh_buydip import GHBuyTheDip
from util.gh_openclose import GHBuyOpenSellClose, GHBuyCloseSellOpen
from util.jh_std_dev import JHReactiveStdDev_tuned
from util.jh_simple import \
    JHBandWagon,\
    JHLongHaul,\
    JHReverseMomentum,\
    JHReverseMomentum_tuned
from util.jh_minmax import JHMinMax
from util.jh_refmodels import JHOmniscientMinMax, JHRandom


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


# ----- functions for plotting/etc
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
        show_plot=True,
        save_fig=False,
):
    xx = df_w_dates.index.values
    
    price_color = 'tab:blue'
    value_color = 'tab:orange'

    fig, [ax1, ax3] = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
        sharex=True,
    )
    
    # ----- plot price
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
    ax1.tick_params(
        axis='x',          # modify the x-axis
        which='both',      # apply to both major and minor ticks
        bottom=False,      # turn off ticks along the bottom edge
        top=False,         # turn off ticks along the top edge
        labelbottom=False  # turn off labels along the bottom edge
    )
    
    # ----- plot decisions
    ax2 = ax1.twinx()
    ax2.axhline(0, linestyle="--", color=".5")
    ax2.plot(
        xx,
        model.trades,
        color=value_color,
        alpha=.5,
        linestyle='',
        marker='.',
    )
    ax2.set_ylabel('Decisions')
    ax2.spines['right'].set_color(value_color)
    ax2.yaxis.label.set_color(value_color)
    ax2.tick_params(
        axis='y',
        which='both',
        color=value_color,
        labelcolor=value_color,
    )
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # add title
    model_name = type(model).__name__
    net_perf = model.get_net_value()
    net_perf = f'{"-" if net_perf < 0 else ""}${abs(net_perf):.2f}'
    title = [
        f'{model_name} model - Prices and Decisions',
        f'Net Financial Performance: {net_perf}',
    ]
    if time_perf_ms is not None:
        title.append(f'Time Performance: {time_perf_ms:.3f} ms')
    plt.title(' \n '.join(title))
    
    # ----- plot net value
    ax3.plot(
        xx,
        model.net_values,
        color='black',
    )
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Net Value')
    
    
    # ----- render and save
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

# TODO: Reconcile colors with plot_decisions? Black is used for input data
#       here, but plot_decisions uses it for net model value.
def plot_comp(
        fin_comp_data,
        incl_sds=False,
        show_plot=True,
        save_fig=False,
):
    fin_comp_df = None
    for row in fin_comp_data:
        _df = pd.DataFrame(
            row['data'],
            columns=['Net Value'],
            index=df_w_dates.index,
        )
        _df['Model'] = row['name']
        fin_comp_df = _df if fin_comp_df is None\
            else pd.concat([fin_comp_df, _df])
    fin_comp_cols = [*fin_comp_df.columns.values]
    fin_comp_cols.insert(0, 'Time')
    fin_comp_df.reset_index(inplace=True)
    fin_comp_df.columns = fin_comp_cols
    
    # NOTE: e.g. to convert timestamp to numeric for linreg
    # fin_comp_df['time'] = fin_comp_df['time'].apply(lambda x: x.value)
    
    tab10 = sns.color_palette("tab10")
    (r, g, b) = tab10[0]
    market_color = '#%02x%02x%02x' % (round(r*255), round(g*255), round(b*255))
    market_color = 'black'
    fig, [ax1, ax2] = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
        sharex=True,
    )
    sns.lineplot(
        data=fin_comp_df,
        x='Time',
        y='Net Value',
        hue='Model',
        # palette=tab10[1:len(fin_comp_data)+1],
        palette=tab10[:len(fin_comp_data)],
        ax=ax1,
    )
    ax1.tick_params(
        axis='x',          # modify the x-axis
        which='both',      # apply to both major and minor ticks
        bottom=False,      # turn off ticks along the bottom edge
        top=False,         # turn off ticks along the top edge
        labelbottom=False  # turn off labels along the bottom edge
    )
    
    
    # ----- add delta
    # fin_comp_df = pd.concat(fin_comp)
    # fin_comp_df.sort_values(['trend','date'], ascending=[True,True], inplace=True)
    # df['shift'] = df.groupby('group')['value'].shift()
    # df['diff'] = df['value'] - df['shift']
    # df = df[['date','group','value','diff']]
    
    # calculate deltas
    # for row in fin_comp:
    #     row['delta'] = np.concatenate(
    #         (
    #             np.zeros((1,)),
    #             np.diff(model.net_values),
    #         ),
    #     )
    
    if incl_sds:
        deltas = np.diff(one_dim)
        for i, delta_sds in enumerate(deltas / one_dim.std()):
            color = 'tab:blue' if delta_sds > 0 else 'tab:orange'
            [time1, time2] = df_w_dates.index.values[i:i+2]
            ax2.axvspan(
                time1,
                time2,
                color=color,
                alpha=abs(np.clip(delta_sds * 2, -1, 1))
            )
    ax2.plot(
        df_w_dates.index.values,
        one_dim,
        color=market_color,
    )
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Price')
    fig.suptitle('Financial Performance - A Few Models')
    plt.tight_layout()
    plt.tight_layout()
    if save_fig:
        plt.savefig(
            'figs/financial_comparison.png',
            dpi=300,
            bbox_inches='tight'
        )
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    plt.show()


# ----- main execution
def main():
    # TODO delete these argument override
    # include_plots = True
    # save_figs = True
    comp_models = [model_type.__name__ for model_type in [
        GHBuyCloseSellOpen,
        JHMinMax,
        JHOmniscientMinMax,
        JHRandom,
    ]]
    fin_comp_data = []
    
    results = []
    for model_type in [
            GHBuyCloseSellOpen,
            GHBuyOpenSellClose,
            GHBuyTheDip,
            JHBandWagon,
            JHLongHaul,
            JHMinMax,
            JHOmniscientMinMax,
            JHRandom,
            JHReverseMomentum,
            JHReverseMomentum_tuned,
            JHReactiveStdDev_tuned,
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
        fin_perf = model.get_net_value()
        results.append([
            model_name,
            model,
            fin_perf,
            time_perf_ms,
        ])
        if model_name in comp_models:
            fin_comp_data.append({
                'name': model_name,
                'data': model.net_values,
            })
        
        if include_plots:
            plot_decisions(
                model,
                time_perf_ms=time_perf_ms,
                save_fig=save_figs,
            )
    
    results = np.array(results)
    
    if include_plots:
        plot_comp(
            fin_comp_data,
            incl_sds=False,
            save_fig=save_figs,
        )
    
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
            f'average time performance over {time_perf_iter} '
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