#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 23:39:39 2022

@author: jhautala
"""

import argparse
import timeit
import tracemalloc # TODO: figure out how we might automate memory analysis?
import getpass

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.ioff() # disable interactive plotting

# internal
from util.model import default_budget
from util.data import load_data, perf_filename
from util import spy, ndaq

# models
# TODO: find a way to simplify all these imports? seems like a lot
from util.co_vol_and_price import COLowAndSlow
from util.gh_buydip import GHBuyTheDip
from util.gh_openclose import GHBuyOpenSellClose, GHBuyCloseSellOpen
from util.jh_minmax import JHMinMax
from util.jh_norm_prob import JHNormProb,\
    JHNormProb_tuned,\
    JHNormThresh,\
    JHNormThresh_tuned
from util.jh_std_dev import JHReactiveStdDev, JHReactiveStdDev_tuned
from util.jh_simple import \
    JHBandWagon,\
    RefLongHaul,\
    JHReverseMomentum,\
    JHReverseMomentum_tuned
from util.mm_buytrendneg import MMbuytrendneg
from util.mm_buytrendpos import MMbuytrendpos
from util.refmodels import RefOmniscientMinMax,\
    RefOmniscientMinMaxNdaq,\
    RefRandom,\
    RefRandomProp


# ----- constants
savefig_kwargs = {
    'dpi': 300,
    'bbox_inches': 'tight',
}


# ----- arg parsing
parser = argparse.ArgumentParser(
    prog = 'CS5800 trends - evaluate models',
    description = 'Compare different trading models',
    epilog = 'Text at the bottom of help',
)
parser.add_argument(
    '--data-source',
    metavar='',
    type=str,
    default='SPY',
    help='which trend to use as the data source'
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
    action=argparse.BooleanOptionalAction,
    default=False,
    help='option to display plots of prices vs decisions'
)
parser.add_argument(
    '--save-figs',
    type=bool,
    action=argparse.BooleanOptionalAction,
    default=False,
    help='option to save plots of prices vs decisions'
)
parser.add_argument(
    '--update-perf',
    type=bool,
    action=argparse.BooleanOptionalAction,
    default=False,
    help='option to update performance data CSV'
)

# - extract args
args=parser.parse_args()
time_perf_iter = args.time_performance_iterations
include_plots = args.include_plots
save_figs = args.save_figs
update_perf = args.update_perf
trend = load_data(args.data_source)

# TODO delete these argument override
# trend = ndaq
# include_plots = True
# save_figs = True
# time_perf_iter = 100
# update_perf = True

# - establish models to compare
omniscient_model = RefOmniscientMinMaxNdaq\
    if trend == ndaq\
    else RefOmniscientMinMax

comp_models = [
    omniscient_model,
    RefLongHaul,
    RefRandomProp,
    COLowAndSlow,
    GHBuyTheDip,
    JHMinMax,
    # JHNormProb,
    # JHNormProb_tuned,
    # JHNormThresh,
    # JHNormThresh_tuned,
    # JHReactiveStdDev,
    JHReactiveStdDev_tuned,
    # MMbuytrendpos,
    MMbuytrendneg,
]
model_names = [model_type.__name__ for model_type in comp_models]

def get_palette(cols=model_names):
    '''
    This is kind of a wonky function for consistently assigning colors to models.
    '''
    tab10 = get_tab10()
    per_col = {}
    for i,col in enumerate(model_names):
        per_col[col] = tab10[i]
    return [per_col[col] for col in cols]


# ----- functions for plotting/etc
def get_tab10():
    tab10 = plt.get_cmap('tab10')(np.arange(10, dtype=int))
    return [
        '#%02x%02x%02x' % (round(r*255), round(g*255), round(b*255))\
            for (r,g,b,a) in tab10
    ]

def evaluate_model(
        trend,
        model_type,
        budget=default_budget,
        skip_perf=False,
):
    n = trend.two_dim.shape[0]
    model = model_type(budget)
    for i in range(1, n+1):
        model.evaluate(trend.two_dim[:i,:].copy())
    return model

def plot_decisions(
        trend,
        model,
        time_perf_ms=None,
        show_plot=True,
        save_fig=False,
):
    xx = trend.df_w_dates.index.values
    
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
        trend.one_dim,
        c=price_color,
        alpha=.5,
    )
    ax1.set_ylabel(f'{trend.name} Price (USD)')
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
        f'{model_name} model - Prices and Decisions vs {trend.name}',
        f'Net Financial Performance: {net_perf}',
    ]
    if time_perf_ms is not None:
        title.append(f'Time Performance: {time_perf_ms:.3f} ms')
    plt.title(' \n '.join(title))
    
    # ----- plot net value
    ax3.plot(
        xx,
        model.net_values,
        color=price_color,
        alpha=.5,
    )
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Net Value (USD)')
    ax3.yaxis.label.set_color(price_color)
    ax3.spines['left'].set_color(price_color)
    ax3.tick_params(
        axis='y',
        which='both',
        color=price_color,
        labelcolor=price_color,
    )
    
    # ----- plot profit
    sell_values = []
    total_invested = 0
    shares_held = 0
    for i, x in enumerate(model.trades):
        curr_price = trend.one_dim[i]
        if x > 0:
            total_invested += curr_price * x
            shares_held += x
        elif x < 0:
            avg_price = total_invested/shares_held
            total_invested += avg_price * x
            shares_held += x
            sell_values.append(-x * (curr_price - avg_price))
        if x >= 0:
            prev = sell_values[-1] if len(sell_values) else 0
            sell_values.append(prev)
    
    ax4 = ax3.twinx()
    ax4.axhline(0, linestyle="--", color=".5")
    ax4.plot(
        xx,
        sell_values,
        color=value_color,
        alpha=.5,
    )
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Profit (USD)')
    ax4.yaxis.label.set_color(value_color)
    ax4.spines['right'].set_color(value_color)
    ax4.tick_params(
        axis='y',
        which='both',
        color=value_color,
        labelcolor=value_color,
    )
    
    # ----- render and save
    plt.tight_layout()
    if save_fig:
        plt.savefig(
            f'figs/{trend.name}_price_vs_decisions_{model_name}.png',
            **savefig_kwargs,
        )
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

# TODO: Reconcile colors with plot_decisions? Black is used for input data
#       here, but plot_decisions uses it for net model value.
def plot_comp(
        trend,
        fin_comp_data,
        incl_sds=False,
        show_plot=True,
        save_fig=False,
):
    fin_comp_df = None
    for row in fin_comp_data:
        _df = pd.DataFrame(
            row['data'],
            columns=['Net Value (USD)'],
            index=trend.df_w_dates.index,
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
    
    fig, [ax1, ax2] = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
        sharex=True,
    )
    
    sns.lineplot(
        data=fin_comp_df,
        x='Time',
        y='Net Value (USD)',
        hue='Model',
        # palette=tab10[1:len(fin_comp_data)+1],
        palette=get_palette(fin_comp_df['Model'].unique()),
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
    
    # # TODO: delete this; it's not fast nor particularly readable...
    # if incl_sds:
    #     deltas = np.diff(trend.one_dim)
    #     for i, delta_sds in enumerate(deltas / trend.one_dim.std()):
    #         color = 'tab:blue' if delta_sds > 0 else 'tab:orange'
    #         [time1, time2] = trend.df_w_dates.index.values[i:i+2]
    #         ax2.axvspan(
    #             time1,
    #             time2,
    #             color=color,
    #             alpha=abs(np.clip(delta_sds * 2, -1, 1))
    #         )
    
    price_color = 'tab:blue'
    price_label = f'{trend.name} Price (USD)'
    price_line = ax2.plot(
        trend.df_w_dates.index.values,
        trend.one_dim,
        color=price_color,
        # alpha=.5,
        label=price_label,
    )
    ax2.set_xlabel('Time')
    ax2.set_ylabel(price_label)
    ax2.spines['left'].set_color(price_color)
    ax2.yaxis.label.set_color(price_color)
    ax2.tick_params(
        axis='y',
        which='both',
        color=price_color,
        labelcolor=price_color,
    )
    
    # plot volume
    tmp = trend.two_dim[:,1]
    tmp = tmp[~np.isnan(tmp)]
    vols = np.empty(trend.one_dim.shape, dtype=trend.one_dim.dtype)
    vols[0::2] = tmp
    vols[1::2] = tmp
    
    volume_color = 'tab:gray'
    volume_label = f'{trend.name} Volume (USD)'
    ax3 = ax2.twinx()
    volume_line = ax3.plot(
        trend.df_w_dates.index.values,
        vols,
        color=volume_color,
        alpha=.5,
        label=volume_label,
    )
    ax3.set_ylabel(volume_label)
    ax3.spines['right'].set_color(volume_color)
    ax3.yaxis.label.set_color(volume_color)
    ax3.tick_params(
        axis='y',
        which='both',
        color=volume_color,
        labelcolor=volume_color,
    )
    
    # NOTE: This is a hack to prevent the legend occluding the Volume trend
    #       for SPY. We add 7% to the upper bound of the y-axis.
    (ylim_0, ylim_1) = ax3.get_ylim()
    ax3.set_ylim(ylim_0, ylim_0 + (ylim_1 - ylim_0) * 1.07)
    
    plt.legend(
        [*price_line, *volume_line],
        (price_label, volume_label),
        # bbox_to_anchor=(.1, 1),
        loc='upper left',
        # borderaxespad=0.,
    )
    fig.suptitle(f'Financial Performance on {trend.name} - {len(fin_comp_data)} Models')
    plt.tight_layout()
    if save_fig:
        plt.savefig(
            f'figs/{trend.name}_financial_comparison_{len(fin_comp_data)}_models.png',
            **savefig_kwargs,
        )
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def plot_rank(
        trend,
        results,
        time_perf_iter,
        show_plot=True,
        save_fig=False,
):
    _df = pd.DataFrame(
        results[:,[0,2,3]],
        columns=['Model', 'Net Value (USD)', 'Avg Time (ms)'],
    )
    _df = _df[_df['Model'].isin(model_names)]
    _df['const'] = 0
    
    if time_perf_iter:
        fig, [ax1, ax2] = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(12, 8),
            sharex=False,
        )
    else:
        fig, ax1 = plt.subplots(
            figsize=(12, 6),
        )
        ax2 = None
    for (ax, asc, col, title) in [
            (
                ax1,
                False,
                'Net Value (USD)',
                f'Financial Performance on {trend.name}',
            ),
            (
                ax2,
                True,
                'Avg Time (ms)',
                f'Time Performance on {trend.name} - {time_perf_iter} Iterations',
            ),
    ]:
        if ax is None: continue;
        _df = _df.sort_values(by=col, ascending=asc)
        sns.barplot(
            data=_df,
            x=col,
            y='Model',
            # hue='Model',
            palette=get_palette(_df['Model'].unique()),
            orient='h',
            ax=ax,
        )
        ax.title.set_text(title)
        ax.set_title(title, fontdict={'fontsize': 15 })
        ax.set_ylabel('')
        # plt.xticks(rotation = 90)
        # ax1.set_xlabel('')
        # ax1.tick_params(
        #     axis='x',          # modify the x-axis
        #     which='both',      # apply to both major and minor ticks
        #     bottom=False,      # turn off ticks along the bottom edge
        #     top=False,         # turn off ticks along the top edge
        #     labelbottom=False  # turn off labels along the bottom edge
        # )
    plt.tight_layout()
    plt.subplots_adjust(
        left=None,
        bottom=None,
        right=None,
        top=None,
        wspace=None,
        hspace=.33,
    )

    if save_fig:
        plt.savefig(
            f'figs/{trend.name}_model_rankings.png',
            **savefig_kwargs,
        )
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ----- main execution
def main():
    fin_comp_data = []
    results = []
    for (model_type, model_name) in zip(comp_models, model_names):
        # print(f'trying {model_name}')
        
        # measure time performance
        if time_perf_iter == 0:
            time_perf_ms = None
        else:
            # timeit*1000 to convert time to milliseconds
            time_perf_ms = timeit.timeit(
                lambda: evaluate_model(trend, model_type),
                number=time_perf_iter,
            )*1000/time_perf_iter
        
        # measure financial performance
        model = evaluate_model(trend, model_type)
        fin_perf = model.get_net_value()
        
        # save results
        results.append([
            model_name,
            model,
            fin_perf,
            time_perf_ms,
        ])
        if model_name in model_names:
            fin_comp_data.append({
                'name': model_name,
                'data': model.net_values,
            })
        
        # plot decisions and net value
        if include_plots or save_figs:
            plot_decisions(
                trend,
                model,
                time_perf_ms=time_perf_ms,
                show_plot=include_plots,
                save_fig=save_figs,
            )
    
    results = np.array(results)
    
    
    # update the performance data CSV
    if update_perf:
        username = getpass.getuser()
        result_df = pd.read_csv(perf_filename)
        
        # remove prior entries for the current user/dataset/models
        user_mask = result_df['username'] == username
        dataset_mask = result_df['dataset'] == trend.name
        model_mask = result_df['model'].isin(results[:,0])
        others_df = result_df.drop(
            result_df[user_mask & dataset_mask & model_mask].index,
        )
        
        # create new df from current execution
        column_shape = (results.shape[0], 1)
        curr_df = pd.DataFrame(
            np.concatenate(
                [
                    np.full(column_shape, trend.name),
                    results[:,[0,2,3]],
                    np.full(column_shape, time_perf_iter),
                    np.full(column_shape, username),
                ],
                axis=1,
            ),
            columns=[
                'dataset',
                'model',
                'financial_performance',
                'time_performance',
                'iterations',
                'username',
            ],
        )
        
        # concatenate and save the updated DataFrame
        result_df = pd.concat([others_df, curr_df])
        result_df.to_csv(perf_filename, index=False)
    
    # plot performance comparisons
    if include_plots or save_figs:
        # financial trends
        plot_comp(
            trend,
            fin_comp_data,
            show_plot=include_plots,
            save_fig=save_figs,
        )
        
        # performance ranking
        plot_rank(
            trend,
            results,
            time_perf_iter,
            show_plot=include_plots,
            save_fig=save_figs,
        )
    
    print(f'financial performance on {trend.name}:')
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