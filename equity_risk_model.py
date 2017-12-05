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
fctr_size = np.log(pri*shr)
fctr_size_norm = fctr_size.sub(fctr_size.mean(axis=1), axis=0).div(fctr_size.std(axis=1), axis=0)

# Value factor
fctr_valu  = pd.rolling_sum(eps, window=12, min_periods=None) / pri
fctr_valu_norm = fctr_valu.sub(fctr_valu.mean(axis=1), axis=0).div(fctr_valu.std(axis=1), axis=0)

# Momentum factor
fctr_mntm = pri.pct_change(periods=12)
fctr_mntm_norm = fctr_mntm.sub(fctr_mntm.mean(axis=1), axis=0).div(fctr_mntm.std(axis=1), axis=0)

# Stack factors
stack_pret = pd.DataFrame((pri.pct_change(periods=1) - np.sum(wgt * pri.pct_change(periods=1))).stack())
stack_fctr_size_norm = pd.DataFrame(fctr_size_norm.stack())
stack_fctr_valu_norm = pd.DataFrame(fctr_valu_norm.stack())
stack_fctr_mntm_norm = pd.DataFrame(fctr_mntm_norm.stack())
stack = pd.concat([stack_pret, stack_fctr_size_norm, stack_fctr_valu_norm, stack_fctr_mntm_norm], axis=1)
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
univ_fctr_cvar = np.cov(univ_regr_beta.T)
univ_fctr_corr = np.corrcoef(univ_regr_beta.T)

univ_regr_rsid_std = np.std(univ_regr_rsid, axis=0) * np.sqrt(12)

# Factor exposures
univ_norm_size_expr = fctr_size_norm.loc[DATE]
univ_norm_valu_expr = fctr_valu_norm.loc[DATE]
univ_norm_mntm_expr = fctr_mntm_norm.loc[DATE]
univ_norm_fctr_expr = np.array(pd.concat([univ_norm_size_expr, univ_norm_valu_expr, univ_norm_mntm_expr], axis=1))

# Factor, specific, and total risk
univ_fctr_risk = np.dot(np.dot(univ_norm_fctr_expr, univ_fctr_cvar), univ_norm_fctr_expr.T)
univ_spec_risk = np.diag(univ_regr_rsid_std**2)
univ_totl_risk = univ_fctr_risk + univ_spec_risk

#==============================================================================
# Portfolio Risk
#==============================================================================

# Total risk
port_fctr_risk_totl = np.sqrt(np.dot(np.dot(port.T, univ_fctr_risk), port))
port_spec_risk_totl = np.sqrt(np.dot(np.dot(port.T, univ_spec_risk), port))
port_totl_risk_totl = np.sqrt(np.dot(np.dot(port.T, univ_totl_risk), port))

# Marginal risk
port_fctr_risk_marg = np.dot(univ_fctr_risk, port) / port_fctr_risk_totl / 100
port_spec_risk_marg = np.dot(univ_spec_risk, port) / port_spec_risk_totl / 100
port_totl_risk_marg = np.dot(univ_totl_risk, port) / port_totl_risk_totl / 100

# Contribution to risk
port_fctr_risk_cntr = port * port_fctr_risk_marg * 100
port_spec_risk_cntr = port * port_spec_risk_marg * 100
port_totl_risk_cntr = port * port_totl_risk_marg * 100

# Percent contribution to total risk
port_fctr_risk_pct_cntr = (port_fctr_risk_totl**2 / port_totl_risk_totl**2) * port_fctr_risk_cntr / np.sum(port_fctr_risk_cntr)
port_spec_risk_pct_cntr = (port_spec_risk_totl**2 / port_totl_risk_totl**2) * port_spec_risk_cntr / np.sum(port_spec_risk_cntr)
port_totl_risk_pct_cntr = (port_totl_risk_totl**2 / port_totl_risk_totl**2) * port_totl_risk_cntr / np.sum(port_totl_risk_cntr)

#==============================================================================
# Benchmark Risk
#==============================================================================

# Total risk
bench_fctr_risk_totl = np.sqrt(np.dot(np.dot(bench.T, univ_fctr_risk), bench))
bench_spec_risk_totl = np.sqrt(np.dot(np.dot(bench.T, univ_spec_risk), bench))
bench_totl_risk_totl = np.sqrt(np.dot(np.dot(bench.T, univ_totl_risk), bench))

# Marginal risk
bench_fctr_risk_marg = np.dot(univ_fctr_risk, bench) / bench_fctr_risk_totl / 100
bench_spec_risk_marg = np.dot(univ_spec_risk, bench) / bench_spec_risk_totl / 100
bench_totl_risk_marg = np.dot(univ_totl_risk, bench) / bench_totl_risk_totl / 100

# Contribution to risk
bench_fctr_risk_cntr = bench * bench_fctr_risk_marg * 100
bench_spec_risk_cntr = bench * bench_spec_risk_marg * 100
bench_totl_risk_cntr = bench * bench_totl_risk_marg * 100

# Percent contribution to total risk
bench_fctr_risk_pct_cntr = (bench_fctr_risk_totl**2 / bench_totl_risk_totl**2) * bench_fctr_risk_cntr / np.sum(bench_fctr_risk_cntr)
bench_spec_risk_pct_cntr = (bench_spec_risk_totl**2 / bench_totl_risk_totl**2) * bench_spec_risk_cntr / np.sum(bench_spec_risk_cntr)
bench_totl_risk_pct_cntr = (bench_totl_risk_totl**2 / bench_totl_risk_totl**2) * bench_totl_risk_cntr / np.sum(bench_totl_risk_cntr)

#==============================================================================
# Active Risk
#==============================================================================

# Total risk
active_fctr_risk_totl = np.sqrt(np.dot(np.dot(active.T, univ_fctr_risk), active))
active_spec_risk_totl = np.sqrt(np.dot(np.dot(active.T, univ_spec_risk), active))
active_totl_risk_totl = np.sqrt(np.dot(np.dot(active.T, univ_totl_risk), active))

# Marginal risk
active_fctr_risk_marg = np.dot(univ_fctr_risk, active) / active_fctr_risk_totl / 100
active_spec_risk_marg = np.dot(univ_spec_risk, active) / active_spec_risk_totl / 100
active_totl_risk_marg = np.dot(univ_totl_risk, active) / active_totl_risk_totl / 100

# Contribution to risk
active_fctr_risk_cntr = active * active_fctr_risk_marg * 100
active_spec_risk_cntr = active * active_spec_risk_marg * 100
active_totl_risk_cntr = active * active_totl_risk_marg * 100

# Percent contribution to total risk
active_fctr_risk_pct_cntr = (active_fctr_risk_totl**2 / active_totl_risk_totl**2) * active_fctr_risk_cntr / np.sum(active_fctr_risk_cntr)
active_spec_risk_pct_cntr = (active_spec_risk_totl**2 / active_totl_risk_totl**2) * active_spec_risk_cntr / np.sum(active_spec_risk_cntr)
active_totl_risk_pct_cntr = (active_totl_risk_totl**2 / active_totl_risk_totl**2) * active_totl_risk_cntr / np.sum(active_totl_risk_cntr)

#==============================================================================
# Factor risk decomposition
#==============================================================================

# Security Factor Exposures
port_fctr_expr = np.dot(port.T, univ_norm_fctr_expr)
bench_fctr_expr = np.dot(bench.T, univ_norm_fctr_expr)
active_fctr_expr = np.dot(active.T, univ_norm_fctr_expr)

# Factor Risk Decomposition
port_fctr_dcmp_marg = np.dot(univ_fctr_cvar, port_fctr_expr.T) / port_fctr_risk_totl / 100
port_fctr_dcmp_cntr = port_fctr_expr.T * port_fctr_dcmp_marg * 100
port_fctr_dcmp_pct_cntr = port_fctr_dcmp_cntr / port_fctr_risk_totl

bench_fctr_dcmp_marg = np.dot(univ_fctr_cvar, bench_fctr_expr.T) / bench_fctr_risk_totl / 100
bench_fctr_dcmp_cntr = bench_fctr_expr.T * bench_fctr_dcmp_marg * 100
bench_fctr_dcmp_pct_cntr = bench_fctr_dcmp_cntr / bench_fctr_risk_totl

active_fctr_dcmp_marg = np.dot(univ_fctr_cvar, active_fctr_expr.T) / active_fctr_risk_totl / 100
active_fctr_dcmp_cntr = active_fctr_expr.T * active_fctr_dcmp_marg * 100
active_fctr_dcmp_pct_cntr = active_fctr_dcmp_cntr / active_fctr_risk_totl

#==============================================================================
# The End
#==============================================================================