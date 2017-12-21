## Equity Risk Model

This is a fundamentally-based [equity risk model](/equity_risk_model.py) that performs factor risk decomposition and single stock risk attribution. The model assumes that only three risk factors&mdash;Size, Value, Momentum&mdash;are relevant and that only ten securities&mdash;AAPL, BA, CAT, DIS, EBAY, F, GOOGL, HOG, IBM, JPM&mdash;are in our universe.

*Note: This model is a drastic oversimplification of equity risk and is intended purely for instructional purposes only. For the United States market, a more realistic model would typically include several more risk factors and thousands of securities.*

The three factors in this model are:

1. **Size**, which is defined as the natural log of the market capitalization
2. **Value**, which is defined as the trailing 12-month earnings yield
3. **Momentum**, which is defined as price change from 12 months ago to 1 month ago

### How It Works

1. Enter the weights of the portfolio and benchmark in their respective CSV files
2. Run the [equity risk model](/equity_risk_model.py)
3. Assign the **portfolio**, **benchmark**, and **active** portfolios as RiskModel objects, **p**, **b**, and **a**, respectively:
```
p = RiskModel(portfolio)
```
```
b = RiskModel(benchmark)
```
```
a = RiskModel(active)
```
4. Calculate a risk measure by executing its repsective method. For example, the total factor risk of the active portfolio, **a**, is called as:

```
a.factor_risk_total()
```

### Risk Analysis

For the portfolio, benchmark, and active portfolios, the total risk is defined as the square root of the sum of the squared factor risk and squared specific risk. Risk is further decomposed at both the stock-level and factor-level. And, for each level, we calculate the following three risk measures:

- Marginal contribution to risk
- Contribution to risk
- Percent contribution to risk

These measures are available as the following methods:

**Portfolio-level**

| # | Method | Level | Definition |
| :---: | :--- | :--- | :--- |
|1| factor_risk_total | Portfolio | Total factor risk |
|2| specific_risk_total | Portfolio | Total specific risk |
|3| total_risk_total | Portfolio | Total risk |

**Stock-level**

| # | Method | Level | Definition |
| :---: | :--- | :--- | :--- |
|1| factor_risk_marginal | Stock | Marginal contribution to factor risk | 
|2| specific_risk_marginal | Stock | Marginal contribution to specific risk | 
|3| total_risk_marginal | Stock | Marginal contribution to total risk | 
|4| factor_risk_contrib | Stock | Contribution to factor risk | 
|5| specific_risk_contrib | Stock | Contribution to specific risk | 
|6| total_risk_contrib | Stock | Contribution to total risk | 
|7| factor_risk_pct_contrib | Stock | Percent contribution to factor risk | 
|8| specific_risk_pct_contrib | Stock | Percent contribution to specific risk | 
|9| total_risk_pct_contrib | Stock | Percent contribution to total risk | 

**Factor-level**

| # | Method | Level | Definition |
| :---: | :--- | :--- | :--- |
|1| factor_decomp_marginal | Factor | Marginal contribution to risk |
|2| factor_decomp_contrib | Factor | Contribution to risk | 
|3| factor_decomp_pct_contrib | Factor | Percent contribution to risk | 

### License

This project is licensed under the [MIT License](/LICENSE).