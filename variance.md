## Variance-based Models

We thought it might be worthwhile to calculate sample statistics for each snapshot. If we want to use the sample mean we might consider that the data is non-stationary and liable to trend, so we might want to apply a sliding window to limit the influence of older data. For simplicity, the initial model takes a single `window` parameter to configure the number of points to include in sample stats calculations. Since each sample is a strict subset of each subsequent sample, we record sufficient statistics for the sliding window instead of recalculating everything for each iteration. This is similar to the rolling hash used in the Rabin-Karp subtring matching algorithm.

### Scaled by Standard Deviation

We started with the same strategy as our most successful preliminary simple model, the ReactiveGreedy (or the "Reverse Momentum") strategy. If we want to scale our response to match the severity of the difference from the prior price, we might want to use the windowed sample standard deviation (or WSSD). We plotted the number of WSSDs difference from point to point on top of the raw data for a window size of 100:

<img src="figs/std_dev-sd_diffs-price_vs_num SDs from prior.png" width=900>

This seems to show that such a model would get moderately "more excited" around notable dips and increases. It also shows that, at least for this data, the response diminishes over time; this conforms to the common investment advice that a younger person (or model) should take more risks and slow down as they approach retirement age (or the end of the simulation... so to speak).

<img src="figs/std_dev-sd_diffs-price_vs_decision costs.png" width=900>

Since we're already arbitrarily choosing the how to scale the buy/sell amounts, we may as well tune it like a hyperparameter. This model does a lot better with very large values of `shares_per_sd` so we tried some very large values, e.g. `1e19`:

<img src="figs/std_dev-sd_diffs_cheat-price_vs_decision costs.png" width=900>

### Scaled by Normal CDF

We tried another model based on sample statistics over a sliding window. The idea here is that we treat the data like a normal distribution (yes, this is probably far more wrong than just a simplification) and we calculate the probability of seeing a lower price than the current value. We rescale that probability to `[-1, 1]` and multiply by our current balance of cash. This produces a more risk-averse model because larger recent variance will actually depress the decision function.

<img src="figs/std_dev-norm-price_vs_net value.png" width=900>

This seems to keep ahead of "the market", but offers modest returns, compared with some of our models (*only* ~40% over 3.5 years). We added an arbitrary `scale` parameter to apply as a factor to the purchase decision quantity:

<img src="figs/std_dev-norm_moderate-price_vs_decision costs.png" width=900>

This probably exceeds the global monetary suplly, but we probed the parameter space again to see if we can do better. We found this model seems to like large values:

<img src="figs/std_dev-norm_cheat-price_vs_decision costs.png" width=900>

It's still a few orders of magnitude behind, but I might be tempted to entrust it with some of my own money.
