## Equity Risk Model

This is a fundamentally-based equity risk model that performs a three-factor risk decomposition and single-stock risk attribution. 

### Risk Model

As an instructional tool and to simplify the calculations, we assume there are only ten securities in our universe: AAPL, BA, CAT, DIS, EBAY, F, GOOGL, HOG, IBM, JPM. 

Note: This model is purely for instructional purposes only. For the United States market, a more realistic universe would typically consist of thousands of securities.

The three factors in this risk model are size, value, and momentum, and they are defined as follows:

1. **Size** is defined as the natural log of the market cap
2. **Value** is defined as the trailing 12-month earnings yield
3. **Momentum** is defined as price change from 12 months ago to 1 month ago

### How It Works

First, enter the weights of your portfolio and benchmark in their respective CSV files. Then, assign the portfolio, benchmark, and active portfolios as RiskModel objects:

```
p = RiskModel(port)
b = RiskModel(bench)
a = RiskModel(active)
```

### Risk Analysis

The total risk of a portfolio, benchmark, or active portfolio is defined as the square root of the sum of the squared factor risk and squared specific risk. Risk is further decomposed at both the factor-level and stock-level, and for each level we perform three risk calculations:

1. Marginal contribution to risk
2. Contribution to risk
3. Percent contribution to risk

To return a risk calculation, you have to call its method.

For example, the total factor risk of the portfolio, p, is called as:
```
p.factor_risk_total()
```

These are all the methods that are currently available:
- factor_risk_total
- specific_risk_total
- total_risk_total
- factor_risk_marginal
- specific_risk_marginal
- total_risk_marginal
- factor_risk_contrib
- specific_risk_contrib
- total_risk_contrib
- factor_risk_pct_contrib
- specific_risk_pct_contrib
- total_risk_pct_contrib
- factor_decomp_marginal
- factor_decomp_contrib
- factor_decomp_pct_contrib

### License

This project is licensed under the [MIT License](/LICENSE).