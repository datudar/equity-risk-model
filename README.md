## Equity Risk Model

This is a fundamentally-based equity [risk model](/equity_risk_model.py) that performs factor risk decomposition and single stock risk attribution. 

We assume that only three risk factors are relevant (size, value, momentum) and that only ten securities are in our universe (AAPL, BA, CAT, DIS, EBAY, F, GOOGL, HOG, IBM, JPM).

*Note: This model is a drastic oversimplification of equity risk and is intended purely for instructional purposes only. For the United States market, a more realistic model would include several more factors and thousands of securities.*

The three factors in this model are:

1. **Size**, which is defined as the natural log of the market capitalization
2. **Value**, which is defined as the trailing 12-month earnings yield
3. **Momentum**, which is defined as price change from 12 months ago to 1 month ago

### How It Works

1. Enter the weights of the portfolio and benchmark in their respective CSV files.
2. Run the [risk model](/equity_risk_model.py).
3. Assign the **portfolio**, **benchmark**, and **active** portfolios as **RiskModel** objects, **p**, **b**, and **a**, respectively:
```
p = RiskModel(portfolio)
```
```
b = RiskModel(benchmark)
```
```
a = RiskModel(active)
```

### Risk Analysis

The total risk of the portfolio, benchmark, or active portfolio is defined as the square root of the sum of the squared factor risk and squared specific risk. Risk is further decomposed at both the factor-level and stock-level. And, for each level, we calculate the following three risk metrics:

- Marginal contribution to risk
- Contribution to risk
- Percent contribution to risk

To return a risk calculation, you call its respective method. For example, the total factor risk of the portfolio, p, is called as:

```
p.factor_risk_total()
```

### Risk Metrics

The following methods are currently available:

**Portfolio-level**

| # | Name | Method | Level |
| :---: | :--- | :--- | :--- |
|1| Total Factor Risk | factor_risk_total | Portfolio |
|2| Total Specific Risk | specific_risk_total | Portfolio |
|3| Total Risk | total_risk_total | Portfolio |

**Stock-level**

| # | Name | Method | Level |
| :---: | :--- | :--- | :--- |
|1| Marginal Factor Risk | factor_risk_marginal | Stock |
|2| Marginal Specific Risk | specific_risk_marginal | Stock |
|3| Marginal Total Risk | total_risk_marginal | Stock |
|4| Contribution to Factor Risk | factor_risk_contrib | Stock |
|5| Contribution to Specific Risk | specific_risk_contrib | Stock |
|6| Contribution to Total Risk | total_risk_contrib | Stock |
|7| Percent Contribution to Factor Risk | factor_risk_pct_contrib | Stock |
|8| Percent Contribution to Specific Risk | specific_risk_pct_contrib | Stock |
|9| Percent Contribution to Total Risk | total_risk_pct_contrib | Stock |

**Factor-level**

| # | Name | Method | Level |
| :---: | :--- | :--- | :--- |
|1| Marginal Factor Risk | factor_decomp_marginal | Factor |
|2| Contribution to Factor Risk | factor_decomp_contrib | Factor |
|3| Percent Contribution to Factor Risk | factor_decomp_pct_contrib | Factor |

### License

This project is licensed under the [MIT License](/LICENSE).