## Variance-based Models

We thought it might be worthwhile to calculate sample statistics for each snapshot. If we want to use the sample mean we might consider that the data is non-stationary and liable to trend, so we might want to apply a sliding window to limit the influence of older data. For simplicity, the initial model takes a single `window` parameter to configure the number of points to include in sample stats calculations.

We started with the same strategy as our most successful preliminary simple model, the ReactiveGreedy (or the "Reverse Momentum") strategy. If we want to scale our response to match the severity of the difference from the prior price, we might want to use the windowed sample standard deviation (or WSSD). We plotted the number of WSSDs difference from point to point on top of the raw data for a window size of 100:

<img src="figs/std_dev_cheat_price_vs_num SDs from prior.png" width=900>

This seems to show that such a model would get moderately "more excited" around notable dips and increases. It also shows that, at least for this data, the response diminishes over time; this conforms to the common investment advice that a younger person (or model) should take more risks and slow down as they approach retirement age (or the end of the simulation... so to speak).

<img src="figs/std_dev_cheat_price_vs_decision costs.png" width=900>