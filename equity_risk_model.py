#==============================================================================
# Import packages
#==============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#==============================================================================
# Initialization Settings
#==============================================================================

DATE = '6/30/2017'
DIR = 'input'
PORT_FILE = '{0}/portfolio.csv'.format(DIR)
BENCH_FILE = '{0}/benchmark.csv'.format(DIR)
WGT_FILE = '{0}/weight.csv'.format(DIR)
PRI_FILE = '{0}/price.csv'.format(DIR)
SHR_FILE  = '{0}/shrsout.csv'.format(DIR)
EPS_FILE  = '{0}/eps.csv'.format(DIR)

#==============================================================================
# Data import
#==============================================================================

port = pd.read_csv(PORT_FILE, index_col='TICKER', header=0) # Portfolio
bench = pd.read_csv(BENCH_FILE, index_col='TICKER', header=0) # Benchmark
active = port - bench # Active portfolio

wgt = pd.read_csv(WGT_FILE, header=0, index_col='DATE', parse_dates=['DATE'])
pri = pd.read_csv(PRI_FILE, header=0, index_col='DATE', parse_dates=['DATE'])
shr = pd.read_csv(SHR_FILE, header=0, index_col='DATE', parse_dates=['DATE'])
eps = pd.read_csv(EPS_FILE, header=0, index_col='DATE', parse_dates=['DATE'])

#==============================================================================
# Create factors
#==============================================================================

# Size factor
factor_size = np.log(pri*shr)
factor_size_norm = factor_size.sub(factor_size.mean(axis=1), axis=0).div(factor_size.std(axis=1), axis=0)

# Value factor
factor_valu  = pd.rolling_sum(eps, window=12, min_periods=None) / pri
factor_valu_norm = factor_valu.sub(factor_valu.mean(axis=1), axis=0).div(factor_valu.std(axis=1), axis=0)

# Momentum factor
factor_mntm = pri.pct_change(periods=12)
factor_mntm_norm = factor_mntm.sub(factor_mntm.mean(axis=1), axis=0).div(factor_mntm.std(axis=1), axis=0)

# Stack factors
stack_pret = pd.DataFrame((pri.pct_change(periods=1) - np.sum(wgt * pri.pct_change(periods=1))).stack())
stack_factor_size_norm = pd.DataFrame(factor_size_norm.stack())
stack_factor_valu_norm = pd.DataFrame(factor_valu_norm.stack())
stack_factor_mntm_norm = pd.DataFrame(factor_mntm_norm.stack())
stack = pd.concat([stack_pret, stack_factor_size_norm, stack_factor_valu_norm, stack_factor_mntm_norm], axis=1)
stack.columns = ['pret', 'size', 'valu', 'mntm']
stack.sort_index(inplace=True)

#==============================================================================
# Regression analysis to calculate factor betas
#==============================================================================

univ_regr_beta = [] # Factor betas
univ_regr_pred = [] # Predictions
univ_regr_rsid = [] # Residuals

for i in stack.index.levels[0][-24:]:
    y = stack.loc[(i)]['pret']
    X = stack.loc[(i)][['size','valu','mntm']]
    
    model = LinearRegression(fit_intercept=False)
    model.fit(X,y)

    pred = model.predict(X)
    rsid = pred - y

    univ_regr_beta.append(model.coef_)
    univ_regr_pred.append(pred)
    univ_regr_rsid.append(rsid)

univ_regr_beta = np.array(univ_regr_beta)
univ_regr_pred = np.array(univ_regr_pred)
univ_regr_rsid = np.array(univ_regr_rsid)

# Factor covariance/correlation matrix
univ_factor_covar = np.cov(univ_regr_beta.T)
univ_factor_corr = np.corrcoef(univ_regr_beta.T)

univ_regr_rsid_std = np.std(univ_regr_rsid, axis=0) * np.sqrt(12)

# Factor exposures
univ_norm_size_expr = factor_size_norm.loc[DATE]
univ_norm_valu_expr = factor_valu_norm.loc[DATE]
univ_norm_mntm_expr = factor_mntm_norm.loc[DATE]
univ_norm_factor_expr = np.array(pd.concat([univ_norm_size_expr, univ_norm_valu_expr, univ_norm_mntm_expr], axis=1))

# Factor, specific, and total risk
univ_factor_risk = np.dot(np.dot(univ_norm_factor_expr, univ_factor_covar), univ_norm_factor_expr.T)
univ_specific_risk = np.diag(univ_regr_rsid_std**2)
univ_total_risk = univ_factor_risk + univ_specific_risk

#==============================================================================
# Portfolio Risk
#==============================================================================

# Total risk
port_factor_risk_total = np.sqrt(np.dot(np.dot(port.T, univ_factor_risk), port))
port_specific_risk_total = np.sqrt(np.dot(np.dot(port.T, univ_specific_risk), port))
port_total_risk_total = np.sqrt(np.dot(np.dot(port.T, univ_total_risk), port))

# margnlinal risk
port_factor_risk_margnl = np.dot(univ_factor_risk, port) / port_factor_risk_total / 100
port_specific_risk_margnl = np.dot(univ_specific_risk, port) / port_specific_risk_total / 100
port_total_risk_margnl = np.dot(univ_total_risk, port) / port_total_risk_total / 100

# Contribution to risk
port_factor_risk_contrib = port * port_factor_risk_margnl * 100
port_specific_risk_contrib = port * port_specific_risk_margnl * 100
port_total_risk_contrib = port * port_total_risk_margnl * 100

# Percent contribution to total risk
port_factor_risk_pct_contrib = (port_factor_risk_total**2 / port_total_risk_total**2) * port_factor_risk_contrib / np.sum(port_factor_risk_contrib)
port_specific_risk_pct_contrib = (port_specific_risk_total**2 / port_total_risk_total**2) * port_specific_risk_contrib / np.sum(port_specific_risk_contrib)
port_total_risk_pct_contrib = (port_total_risk_total**2 / port_total_risk_total**2) * port_total_risk_contrib / np.sum(port_total_risk_contrib)

#==============================================================================
# Benchmark Risk
#==============================================================================

# Total risk
bench_factor_risk_total = np.sqrt(np.dot(np.dot(bench.T, univ_factor_risk), bench))
bench_specific_risk_total = np.sqrt(np.dot(np.dot(bench.T, univ_specific_risk), bench))
bench_total_risk_total = np.sqrt(np.dot(np.dot(bench.T, univ_total_risk), bench))

# margnlinal risk
bench_factor_risk_margnl = np.dot(univ_factor_risk, bench) / bench_factor_risk_total / 100
bench_specific_risk_margnl = np.dot(univ_specific_risk, bench) / bench_specific_risk_total / 100
bench_total_risk_margnl = np.dot(univ_total_risk, bench) / bench_total_risk_total / 100

# Contribution to risk
bench_factor_risk_contrib = bench * bench_factor_risk_margnl * 100
bench_specific_risk_contrib = bench * bench_specific_risk_margnl * 100
bench_total_risk_contrib = bench * bench_total_risk_margnl * 100

# Percent contribution to total risk
bench_factor_risk_pct_contrib = (bench_factor_risk_total**2 / bench_total_risk_total**2) * bench_factor_risk_contrib / np.sum(bench_factor_risk_contrib)
bench_specific_risk_pct_contrib = (bench_specific_risk_total**2 / bench_total_risk_total**2) * bench_specific_risk_contrib / np.sum(bench_specific_risk_contrib)
bench_total_risk_pct_contrib = (bench_total_risk_total**2 / bench_total_risk_total**2) * bench_total_risk_contrib / np.sum(bench_total_risk_contrib)

#==============================================================================
# Active Risk
#==============================================================================

# Total risk
active_factor_risk_total = np.sqrt(np.dot(np.dot(active.T, univ_factor_risk), active))
active_specific_risk_total = np.sqrt(np.dot(np.dot(active.T, univ_specific_risk), active))
active_total_risk_total = np.sqrt(np.dot(np.dot(active.T, univ_total_risk), active))

# margnlinal risk
active_factor_risk_margnl = np.dot(univ_factor_risk, active) / active_factor_risk_total / 100
active_specific_risk_margnl = np.dot(univ_specific_risk, active) / active_specific_risk_total / 100
active_total_risk_margnl = np.dot(univ_total_risk, active) / active_total_risk_total / 100

# Contribution to risk
active_factor_risk_contrib = active * active_factor_risk_margnl * 100
active_specific_risk_contrib = active * active_specific_risk_margnl * 100
active_total_risk_contrib = active * active_total_risk_margnl * 100

# Percent contribution to total risk
active_factor_risk_pct_contrib = (active_factor_risk_total**2 / active_total_risk_total**2) * active_factor_risk_contrib / np.sum(active_factor_risk_contrib)
active_specific_risk_pct_contrib = (active_specific_risk_total**2 / active_total_risk_total**2) * active_specific_risk_contrib / np.sum(active_specific_risk_contrib)
active_total_risk_pct_contrib = (active_total_risk_total**2 / active_total_risk_total**2) * active_total_risk_contrib / np.sum(active_total_risk_contrib)

#==============================================================================
# Factor risk decomposition
#==============================================================================

# Security Factor Exposures
port_factor_expr = np.dot(port.T, univ_norm_factor_expr)
bench_factor_expr = np.dot(bench.T, univ_norm_factor_expr)
active_factor_expr = np.dot(active.T, univ_norm_factor_expr)

# Factor Risk Decomposition
port_factor_decomp_margnl = np.dot(univ_factor_covar, port_factor_expr.T) / port_factor_risk_total / 100
port_factor_decomp_contrib = port_factor_expr.T * port_factor_decomp_margnl * 100
port_factor_decomp_pct_contrib = port_factor_decomp_contrib / port_factor_risk_total

bench_factor_decomp_margnl = np.dot(univ_factor_covar, bench_factor_expr.T) / bench_factor_risk_total / 100
bench_factor_decomp_contrib = bench_factor_expr.T * bench_factor_decomp_margnl * 100
bench_factor_decomp_pct_contrib = bench_factor_decomp_contrib / bench_factor_risk_total

active_factor_decomp_margnl = np.dot(univ_factor_covar, active_factor_expr.T) / active_factor_risk_total / 100
active_factor_decomp_contrib = active_factor_expr.T * active_factor_decomp_margnl * 100
active_factor_decomp_pct_contrib = active_factor_decomp_contrib / active_factor_risk_total

#==============================================================================
# The End
#==============================================================================