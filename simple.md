## Simple Models

* Random: randomly choose to buy or sell one share per time period

<img src="figs/price_vs_decisions_Random.png" width=900>

* LongHaul: buy as many shares a possible on the first iteration and never sell

<img src="figs/price_vs_decisions_LongHaul.png" width=900>

* OptimisticGreedy: always buy exactly one share per iteration (this ends up being very similar to the LongHaul model)

<img src="figs/price_vs_decisions_OptimisticGreedy.png" width=900>

* BandWagon: buy one share if the current price is greater than the previous, sell if less, and do nothing if equal

<img src="figs/price_vs_decisions_BandWagon.png" width=900>

* ReactiveGreedy: the opposite of the BandWagon model; buy if current is greater than previous, etc.

<img src="figs/price_vs_decisions_ReactiveGreedy.png" width=900>

* OmniscientMinMax: this is not strictly a valid model (also not truly greedy); it uses special knowledge, only buying shares at the global max (spending its entire budget) and only sells at the global minimum (selling all of its shares). It clearly illustrates the opportunity cost of making a low number of decisions.

<img src="figs/price_vs_decisions_OmniscientMinMax.png" width=900>

* BuyTheDip: this is like the ReactiveGreedy model but using a ten day interval. If the current price is lower than the price nine days previously, it will buy a share; if it's higher it will sell a share; and if it's equal it will do nothing.

<img src="figs/price_vs_decisions_BuyTheDip.png" width=900>

* BuyOpenSellClose: buy as many shares as possible each morning and sell all each evening.

<img src="figs/price_vs_decisions_BuyOpenSellClose.png" width=900>

* BuyCloseSellOpen: sell all shares each morning and buy as many as possible each evening.

<img src="figs/price_vs_decisions_BuyCloseSellOpen.png" width=900>

## "Cheat" Models

Since we are only using a single time series, all of our models are indirectly "trained" on "test" data (i.e. any manual adjustments to improve performance are tainted by prescience, a sort of cheat, that could lead to overfitting => loss of generality). For some of our models, we tried parameterizing them and scanning the parameter space for optimal parameters:

* ReactiveGreedy_cheat: This model uses the same logic as ReactiveGreedy (i.e. buying or selling based on negative/positive slope for last two points, respectively) but instead of simply buying/selling exactly one share for each transaction, we tried using other constants. For example we might always try to buy/sell exactly 10 shares. By brute force we determined that, for this data set and initial budget of $10,000, the optimal number of shares was 18.

<img src="figs/price_vs_decisions_ReactiveGreedy_cheat.png" width=900>