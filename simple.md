## Simple Models

* Random: randomly choose to buy or sell one share per time period
* LongHaul: buy as many shares a possible on the first iteration and never sells
* OptimisticGreedy: always buy exactly one share per iteration
* BandWagon: buy one share if the current price is greater than the previous, sell if less, and do nothing if equal
* ReactiveGreedy: the opposite of the BandWagon model
* OmniscientMinMax: this is not strictly a valid model (also not truly greedy); it uses special knowledge, only buying shares at the global max (spending its entire budget) and only sells at the global minimum (selling all of its shares). It clearly illustrates the opportunity cost of making a low number of decisions.
* BuyTheDip: this is like the ReactiveGreedy model but using a ten day interval. If the current price is lower than the price nine days previously, it will buy a share; if it's higher it will sell a share; and if it's equal it will do nothing.
* BuyOpenSellClose: buy as many shares as possible each morning and sell all each evening.
* BuyCloseSellOpen: sell all shares each morning and buy as many as possible each evening.

## "Cheat" Models

Since we are only using a single time series, all of our models are indirectly "trained" on "test" data. For some of our simple models, we tried parameterizing them and found optimal parameters by brute force:

* ReactiveGreedy_cheat: This model uses the same logic as ReactiveGreedy (i.e. buying or selling based on negative/positive derivative for last two points, respectively) but instead of simply buying/selling exactly one share for each transaction, we tried using other constants. For example we might always try to buy/sell exactly 10 shares. By brute force we determined that, for this data set and initial budget of $10,000, the optimal number of shares was 42 (but [of course it was](https://news.mit.edu/2019/answer-life-universe-and-everything-sum-three-cubes-mathematics-0910)).