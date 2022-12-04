## Variance-based Models

We thought it might be worthwhile to calculate sample statistics for each snapshot. If we want to use the sample mean we might consider that the data is non-stationary and liable to trend, so we might want to apply a sliding window to limit the influence of older data. For simplicity, the initial model takes a single `window` parameter to configure the number of points to include in sample stats calculations. Since each sample is a strict subset of each subsequent sample, we record sufficient statistics for the sliding window instead of recalculating everything for each iteration. This is similar to the rolling hash used in the Rabin-Karp subtring matching algorithm.

### Scaled by Standard Deviation

We started with the "reverse momentum" strategy (similar to the ReactiveGreedy model) and tried scaling the response to match the intensity of the difference from the prior price. In such models we might want to use the windowed sample standard deviation (or WSSD). We plotted the number of WSSDs difference from point to point on top of the raw data for a window size of 100.

<img src="figs/std_dev-sd_diffs-price_vs_num SDs from prior.png" width=900>

This seems to show that such a model would get moderately "more excited" around extreme instantaneous dips and increases. In this data set, the point-to-point difference never reaches a single standard deviation, so this strategy would require scaling to produce any net financial performance (positive or negative). By scanning the parameter space, we found it does fairly well at around 300 to 400 and there is a local maximum at 68.6.

<img src="figs/std_dev-sd_diffs_cheat-price_vs_decision costs.png" width=900>

### Scaled by Normal CDF

We tried another model based on sample statistics over a sliding window. The idea here is that we treat the data like a normal distribution (yes, this is probably far more wrong than just a simplification) and we calculate the probability of seeing a lower price than the current value. We rescale that probability to `[-1, 1]` and multiply by our current balance of cash. This produces a more risk-averse model because larger recent variance will actually depress the decision function.

<img src="figs/std_dev-norm-price_vs_decision costs.png" width=900>

This model seems to react too quickly when it sees a spike or drop in price. Again we tried tuning for maximum performance on this dataset and found a marginal increase in performance at a local maximum for `scale=1.496`.

<img src="figs/std_dev-norm_cheat-price_vs_decision costs.png" width=900>

### Provisional Conclusion

This sample statistics may be a useful input to decision making, but we need to work out a way to make it more "patient" (i.e. to not react immediately to spikes/drops in price). Perhaps it should remember how much it spent per share and avoid selling at a loss. We might also consider using different distributions, as opposed to Gaussian, to model the probability.
