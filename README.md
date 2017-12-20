## Equity Risk Model

This is a fundamentally-based equity risk model that performs a three-factor risk decomposition and single-stock risk attribution. It is meant for instructional purposes only and should not be used to model the risk of your portfolio.

### How It Works

As an instructional tool and to simplify the calculations, we assume there are only ten securities in our universe: AAPL, BA, CAT, DIS, EBAY, F, GOOGL, HOG, IBM, JPM. For the United States market, a more realistic universe would typically consist of thousands of stocks.

The weights of your portfolio and benchmark are set in their respective CSV files. Then, assign the portfolio, benchmark, and active portfolios as follows:

'''
p = RiskModel(port)
b = RiskModel(bench)
a = RiskModel(active)
'''

All risk calculations are methods that you can call as follows:

'''
a.factor_risk_total()
'''

### Risk Model

The three factors in the risk model are size, value, and momentum.

#### Factor Definitions

1. The **size** factor is defined as the natural log of the market cap
2. The **value** factor is defined as the earnings yield
3. The **momentum** factor is defined as price change from 12 months ago to 1 month ago

### Risk Decomposition

There are two flavors of risk that we measure: absolute and relative. And for each flavor of risk, we measure factor, specific, and total risks. Total risk is defined as the square root of the sum of the squared factor and squared specific risks.

Risk is further decomposed at both the factor-level and stock-level. The three risk decompositions that we perform are:

1. Marginal contribution to risk
2. Contribution to risk
3. Percent contribution to risk

### License

This project is licensed under the [MIT License](/LICENSE).