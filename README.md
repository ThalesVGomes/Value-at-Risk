# value-at-risk
Python code to calculate the Value at Risk assuming a parametric T-Distribution of returns.
Very useful to monitor your portfolio market risk.

You just need to provide the stock tickers and the desired date to calculate the VaR. Also supports EWMA model calculation.

The code makes large use of numpy vectorization which makes it runs faster. (Normally less than 10 seconds, depends on your hardware)

The bottleneck is the connection with a external database to grab information about the stock prices, but this shouldn't be a problem for normal usage.
If you really need a faster processing i'd suggest that you provide a connection to your own offline database containing the stock prices.
