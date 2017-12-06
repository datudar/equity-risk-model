#==============================================================================
# Import packages
#==============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#==============================================================================
# Initialization Settings
#==============================================================================

DATE = '6/30/2017' # Analysis date
PERIOD = 24 # No. of months to use in calculating factor betas
DIR = 'input' # Input directory
PORT_FILE = '{0}/portfolio.csv'.format(DIR) # Current portfolio
BENCH_FILE = '{0}/benchmark.csv'.format(DIR) # Current benchmark
UNV_FILE = '{0}/universe.csv'.format(DIR) # Current universe
PRI_FILE = '{0}/price.csv'.format(DIR) # Price file
SHR_FILE  = '{0}/shrsout.csv'.format(DIR) # Shares outstanding file
EPS_FILE  = '{0}/eps.csv'.format(DIR) # Earnings per share (EPS) file

#==============================================================================
# Data import
#==============================================================================

port = pd.read_csv(PORT_FILE, index_col='TICKER', header=0) # Portfolio
bench = pd.read_csv(BENCH_FILE, index_col='TICKER', header=0) # Benchmark
active = port - bench # Active portfolio

wgt = pd.read_csv(UNV_FILE, header=0, index_col='DATE', parse_dates=['DATE'])
pri = pd.read_csv(PRI_FILE, header=0, index_col='DATE', parse_dates=['DATE'])
shr = pd.read_csv(SHR_FILE, header=0, index_col='DATE', parse_dates=['DATE'])
eps = pd.read_csv(EPS_FILE, header=0, index_col='DATE', parse_dates=['DATE'])

#==============================================================================
# Create factors
#==============================================================================

# size: 1. Factor: Size
# ====  2. Definition: Natural log of market cap
#       3. Scaling: The values are normalized
#
# valu: 1. Factor: Value
# ====  2. Definition: Earnings yield = EPS / price
#       3. Scaling: The values are normalized
#       4. Note: Earnings is lagged by two months to account for delay in earnings data
#
# mntm: 1. Factor: Price momentum
# ====  2. Definition: Price change from 12 months ago to 1 month ago 
#       3. Scaling: The values are normalized

factor_size = np.log(pri*shr)
factor_size_norm = factor_size.sub(factor_size.mean(axis=1), axis=0).div(factor_size.std(axis=1), axis=0)

factor_valu  = pd.rolling_sum(eps.shift(2), window=12, min_periods=None) / pri
factor_valu_norm = factor_valu.sub(factor_valu.mean(axis=1), axis=0).div(factor_valu.std(axis=1), axis=0)

factor_mntm = pri.shift(1).pct_change(periods=11)
factor_mntm_norm = factor_mntm.sub(factor_mntm.mean(axis=1), axis=0).div(factor_mntm.std(axis=1), axis=0)

# Stack factors to prepare for regression analysis
# Note: factors are lagged by one month so that they are aligned with the
# subsequent monthly price return
stack_pret = pd.DataFrame(pri.pct_change(periods=1).subtract(np.sum(wgt * pri.pct_change(periods=1),axis=1),axis=0).stack())
stack_factor_size_norm = pd.DataFrame(factor_size_norm.shift(1).stack())
stack_factor_valu_norm = pd.DataFrame(factor_valu_norm.shift(1).stack())
stack_factor_mntm_norm = pd.DataFrame(factor_mntm_norm.shift(1).stack())
stack = pd.concat([stack_pret, stack_factor_size_norm, stack_factor_valu_norm, stack_factor_mntm_norm], axis=1)
stack.columns = ['pret', 'size', 'valu', 'mntm']
stack.sort_index(inplace=True)

#==============================================================================
# Regression analysis
#==============================================================================

betas = [] # Factor betas
preds = [] # Predictions
resid = [] # Residuals

for i in stack.index.levels[0][-PERIOD:]:
    y = stack.loc[(i)]['pret']
    X = stack.loc[(i)][['size','valu','mntm']]
    
    model = LinearRegression(fit_intercept=False)
    model.fit(X,y)

    betas.append(model.coef_)
    preds.append(model.predict(X))
    resid.append(y - model.predict(X))

betas = np.array(betas)
preds = np.array(preds)
resid = np.array(resid)

# Factor covariance/correlation matrix
factor_covar = np.cov(betas.T, ddof=1) * 12
factor_corr = np.corrcoef(betas.T)

# Factor exposures
factor_exposure = np.array(pd.concat([factor_size_norm.loc[DATE], 
                                     factor_valu_norm.loc[DATE], 
                                     factor_mntm_norm.loc[DATE],
                                     ],axis=1))

# Factor, specific, and total risk matrices of size NxN
factor_risk = np.dot(np.dot(factor_exposure, factor_covar), factor_exposure.T)
specific_risk = np.diag((np.std(resid, ddof=1, axis=0) * np.sqrt(12))**2)
total_risk = factor_risk + specific_risk

#==============================================================================
# Portfolio risk
#==============================================================================

# Total risk (portfolio-level)
port_factor_risk_total = np.sqrt(np.dot(np.dot(port.T, factor_risk), port))
port_specific_risk_total = np.sqrt(np.dot(np.dot(port.T, specific_risk), port))
port_total_risk_total = np.sqrt(np.dot(np.dot(port.T, total_risk), port))

# Marginal risk (stock-level)
port_factor_risk_marginal = np.dot(factor_risk, port) / port_factor_risk_total / 100
port_specific_risk_marginal = np.dot(specific_risk, port) / port_specific_risk_total / 100
port_total_risk_marginal = np.dot(total_risk, port) / port_total_risk_total / 100

# Contribution to risk (stock-level)
port_factor_risk_contrib = port * port_factor_risk_marginal * 100
port_specific_risk_contrib = port * port_specific_risk_marginal * 100
port_total_risk_contrib = port * port_total_risk_marginal * 100

# Percent contribution to risk (stock-level)
port_factor_risk_pct_contrib = ((port_factor_risk_total**2 / port_total_risk_total**2) * 
                                port_factor_risk_contrib / np.sum(port_factor_risk_contrib))
port_specific_risk_pct_contrib = ((port_specific_risk_total**2 / port_total_risk_total**2) * 
                                  port_specific_risk_contrib / np.sum(port_specific_risk_contrib))
port_total_risk_pct_contrib = ((port_total_risk_total**2 / port_total_risk_total**2) * 
                               port_total_risk_contrib / np.sum(port_total_risk_contrib))

#==============================================================================
# Benchmark risk
#==============================================================================

# Total risk (portfolio-level)
bench_factor_risk_total = np.sqrt(np.dot(np.dot(bench.T, factor_risk), bench))
bench_specific_risk_total = np.sqrt(np.dot(np.dot(bench.T, specific_risk), bench))
bench_total_risk_total = np.sqrt(np.dot(np.dot(bench.T, total_risk), bench))

# Marginal risk
bench_factor_risk_marginal = np.dot(factor_risk, bench) / bench_factor_risk_total / 100
bench_specific_risk_marginal = np.dot(specific_risk, bench) / bench_specific_risk_total / 100
bench_total_risk_marginal = np.dot(total_risk, bench) / bench_total_risk_total / 100

# Contribution to risk
bench_factor_risk_contrib = bench * bench_factor_risk_marginal * 100
bench_specific_risk_contrib = bench * bench_specific_risk_marginal * 100
bench_total_risk_contrib = bench * bench_total_risk_marginal * 100

# Percent contribution to risk
bench_factor_risk_pct_contrib = ((bench_factor_risk_total**2 / bench_total_risk_total**2) * 
                                 bench_factor_risk_contrib / np.sum(bench_factor_risk_contrib))
bench_specific_risk_pct_contrib = ((bench_specific_risk_total**2 / bench_total_risk_total**2) * 
                                   bench_specific_risk_contrib / np.sum(bench_specific_risk_contrib))
bench_total_risk_pct_contrib = ((bench_total_risk_total**2 / bench_total_risk_total**2) * 
                                bench_total_risk_contrib / np.sum(bench_total_risk_contrib))

#==============================================================================
# Active risk
#==============================================================================

# Total risk (portfolio-level)
active_factor_risk_total = np.sqrt(np.dot(np.dot(active.T, factor_risk), active))
active_specific_risk_total = np.sqrt(np.dot(np.dot(active.T, specific_risk), active))
active_total_risk_total = np.sqrt(np.dot(np.dot(active.T, total_risk), active))

# Marginal risk (stock-level)
active_factor_risk_marginal = np.dot(factor_risk, active) / active_factor_risk_total / 100
active_specific_risk_marginal = np.dot(specific_risk, active) / active_specific_risk_total / 100
active_total_risk_marginal = np.dot(total_risk, active) / active_total_risk_total / 100

# Contribution to risk (stock-level)
active_factor_risk_contrib = active * active_factor_risk_marginal * 100
active_specific_risk_contrib = active * active_specific_risk_marginal * 100
active_total_risk_contrib = active * active_total_risk_marginal * 100

# Percent contribution to risk (stock-level)
active_factor_risk_pct_contrib = ((active_factor_risk_total**2 / active_total_risk_total**2) * 
                                  active_factor_risk_contrib / np.sum(active_factor_risk_contrib))
active_specific_risk_pct_contrib = ((active_specific_risk_total**2 / active_total_risk_total**2) * 
                                    active_specific_risk_contrib / np.sum(active_specific_risk_contrib))
active_total_risk_pct_contrib = ((active_total_risk_total**2 / active_total_risk_total**2) * 
                                 active_total_risk_contrib / np.sum(active_total_risk_contrib))

#==============================================================================
# Factor risk decomposition
#==============================================================================

# Security factor exposures
port_factor_exposure = np.dot(port.T, factor_exposure)
bench_factor_exposure = np.dot(bench.T, factor_exposure)
active_factor_exposure = np.dot(active.T, factor_exposure)

# Portfolio factor risk decomposition
port_factor_decomp_marginal = np.dot(factor_covar, port_factor_exposure.T) / port_factor_risk_total / 100
port_factor_decomp_contrib = port_factor_exposure.T * port_factor_decomp_marginal * 100
port_factor_decomp_pct_contrib = port_factor_decomp_contrib / port_factor_risk_total

# Benchmark factor risk decomposition
bench_factor_decomp_marginal = np.dot(factor_covar, bench_factor_exposure.T) / bench_factor_risk_total / 100
bench_factor_decomp_contrib = bench_factor_exposure.T * bench_factor_decomp_marginal * 100
bench_factor_decomp_pct_contrib = bench_factor_decomp_contrib / bench_factor_risk_total

# Active factor risk decomposition
active_factor_decomp_marginal = np.dot(factor_covar, active_factor_exposure.T) / active_factor_risk_total / 100
active_factor_decomp_contrib = active_factor_exposure.T * active_factor_decomp_marginal * 100
active_factor_decomp_pct_contrib = active_factor_decomp_contrib / active_factor_risk_total

#==============================================================================
# The End
#==============================================================================