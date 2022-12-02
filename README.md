# CS5800-trends

We intend to test the financial and computational performance of various automated stock trading algorithms.

## Data

The data used for this project was sourced from the following URL:
[https://finance.yahoo.com/quote/SPY/history/](https://finance.yahoo.com/quote/SPY/history/)

The exact version we are using is also copied into the `data` directory of this repo: [SPY.csv](data/SPY.csv)

## Constraints

In our initial implementation we simplify the problem space by processing successive historic snapshots from a single time series. Each model must implement a simple decision interface where, given each successive historic snapshot, the model must calculate how many shares to buy or sell at the current price. Such algorithms are inherently greedy; they *must* perform online selection.

All models are given the same fixed budget at the start of each simulation and the financial performance is calculated as the `final_balance` - `starting_balance` + `final_shares` * `final_price`.

## Models

* [simple models](simple.md)
* [variance models](variance.md)