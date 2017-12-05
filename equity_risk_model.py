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

WEIGHT_FILE = '{0}/weight.csv'.format(DIR)
PRICE_FILE = '{0}/price.csv'.format(DIR)
SHRSOUT_FILE  = '{0}/shrsout.csv'.format(DIR)
EPS_FILE  = '{0}/eps.csv'.format(DIR)

#==============================================================================
# Data import
#==============================================================================

port = pd.read_csv(PORT_FILE, index_col='TICKER', header=0)
bench = pd.read_csv(BENCH_FILE, index_col='TICKER', header=0)
active = port - bench

weight = pd.read_csv(WEIGHT_FILE, index_col='DATE', header=0)
price = pd.read_csv(PRICE_FILE, index_col='DATE', header=0)
shrsout = pd.read_csv(SHRSOUT_FILE, index_col='DATE', header=0)
eps = pd.read_csv(EPS_FILE, index_col='DATE', header=0)

#pd.to_datetime(weight.index)
weight.index = pd.DatetimeIndex(weight.index)
price.index = pd.DatetimeIndex(price.index)
shrsout.index = pd.DatetimeIndex(shrsout.index)
eps.index = pd.DatetimeIndex(eps.index)

# universe
price_ret = price.pct_change(periods=1)
rel_ret = price_ret - np.sum(weight*price_ret)

# size
size = price * shrsout / 1000000000
log_size = np.log(size)
norm_size = log_size.sub(log_size.mean(axis=1), axis=0).div(log_size.std(axis=1), axis=0)

# value
eps_sum = pd.rolling_sum(eps, window=12, min_periods=None)
eps_yld  = eps_sum / price
norm_valu = eps_yld.sub(eps_yld.mean(axis=1), axis=0).div(eps_yld.std(axis=1), axis=0)

# momentum
mom = price.pct_change(periods=12)
norm_mntm = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1), axis=0)

# stack
stack_rel_ret = pd.DataFrame(rel_ret.stack())
stack_norm_size = pd.DataFrame(norm_size.stack())
stack_norm_valu = pd.DataFrame(norm_valu.stack())
stack_norm_mntm = pd.DataFrame(norm_mntm.stack())

frames = [stack_rel_ret, stack_norm_size, stack_norm_valu, stack_norm_mntm]
result = pd.concat(frames, axis=1)
result.columns = ['RET', 'SIZE', 'VALU', 'MNTM']

result.sort_index(inplace=True)

factor_betas = []
predictions = []
residuals = []

for i in result.index.levels[0][-24:]:
    y = result.loc[(i)].RET
    X = result.loc[(i)][['SIZE','VALU','MNTM']]
    
    model = LinearRegression(fit_intercept=False)
    model.fit(X,y)

    prediction = model.predict(X)
    
    residual = prediction - y

    factor_betas.append(model.coef_)
    predictions.append(prediction)
    residuals.append(residual)

factor_betas = np.array(factor_betas)
predictions = np.array(predictions)
residuals  = np.array(residuals)
 
factor_covar = np.cov(factor_betas.T)
factor_corr = np.corrcoef(factor_betas.T)

residuals_std = np.std(residuals, axis=0) * np.sqrt(12)

# factor exposures
norm_size_exp = norm_size.loc[DATE]
norm_valu_exp = norm_valu.loc[DATE]
norm_mntm_exp = norm_mntm.loc[DATE]

factor_exposures = np.array(pd.concat([norm_size_exp, norm_valu_exp, norm_mntm_exp], axis=1))

factor_risk = np.dot(np.dot(factor_exposures, factor_covar), factor_exposures.T)
specific_risk = np.diag(residuals_std**2)
total_risk = factor_risk + specific_risk

# Portfolio risk
port_factor_risk = np.sqrt(np.dot(np.dot(port.T, factor_risk), port))
port_specific_risk = np.sqrt(np.dot(np.dot(port.T, specific_risk), port))
port_total_risk = np.sqrt(np.dot(np.dot(port.T, total_risk), port))

port_marginal_contrib_factor_risk = np.dot(factor_risk, port) / port_factor_risk / 100
port_marginal_contrib_specific_risk = np.dot(specific_risk, port) / port_specific_risk / 100
port_marginal_contrib_total_risk = np.dot(total_risk, port) / port_total_risk / 100

port_contrib_factor_risk = port * port_marginal_contrib_factor_risk * 100
port_contrib_specific_risk = port * port_marginal_contrib_specific_risk * 100
port_contrib_total_risk = port * port_marginal_contrib_total_risk * 100

port_pct_factor_risk = port_factor_risk**2 / port_total_risk**2
port_pct_specific_risk = port_specific_risk**2 / port_total_risk**2
port_pct_total_risk = port_total_risk**2 / port_total_risk**2

port_pct_contrib_factor_risk = port_pct_factor_risk * port_contrib_factor_risk / np.sum(port_contrib_factor_risk)
port_pct_contrib_specific_risk = port_pct_specific_risk * port_contrib_specific_risk / np.sum(port_contrib_specific_risk)
port_pct_contrib_total_risk = port_pct_total_risk * port_contrib_total_risk / np.sum(port_contrib_total_risk)

# Benchmark risk
bench_factor_risk = np.sqrt(np.dot(np.dot(bench.T, factor_risk), bench))
bench_specific_risk = np.sqrt(np.dot(np.dot(bench.T, specific_risk), bench))
bench_total_risk = np.sqrt(np.dot(np.dot(bench.T, total_risk), bench))

bench_marginal_contrib_factor_risk = np.dot(factor_risk, bench) / bench_factor_risk / 100
bench_marginal_contrib_specific_risk = np.dot(specific_risk, bench) / bench_specific_risk / 100
bench_marginal_contrib_total_risk = np.dot(total_risk, bench) / bench_total_risk / 100

bench_contrib_factor_risk = bench * bench_marginal_contrib_factor_risk * 100
bench_contrib_specific_risk = bench * bench_marginal_contrib_specific_risk * 100
bench_contrib_total_risk = bench * bench_marginal_contrib_total_risk * 100

bench_pct_factor_risk = bench_factor_risk**2 / bench_total_risk**2
bench_pct_specific_risk = bench_specific_risk**2 / bench_total_risk**2
bench_pct_total_risk = bench_total_risk**2 / bench_total_risk**2

bench_pct_contrib_factor_risk = bench_pct_factor_risk * bench_contrib_factor_risk / np.sum(bench_contrib_factor_risk)
bench_pct_contrib_specific_risk = bench_pct_specific_risk * bench_contrib_specific_risk / np.sum(bench_contrib_specific_risk)
bench_pct_contrib_total_risk = bench_pct_total_risk * bench_contrib_total_risk / np.sum(bench_contrib_total_risk)

# Active risk
active_factor_risk = np.sqrt(np.dot(np.dot(active.T, factor_risk), active))
active_specific_risk = np.sqrt(np.dot(np.dot(active.T, specific_risk), active))
active_total_risk = np.sqrt(np.dot(np.dot(active.T, total_risk), active))

active_marginal_contrib_factor_risk = np.dot(factor_risk, active) / active_factor_risk / 100
active_marginal_contrib_specific_risk = np.dot(specific_risk, active) / active_specific_risk / 100
active_marginal_contrib_total_risk = np.dot(total_risk, active) / active_total_risk / 100

active_contrib_factor_risk = active * active_marginal_contrib_factor_risk * 100
active_contrib_specific_risk = active * active_marginal_contrib_specific_risk * 100
active_contrib_total_risk = active * active_marginal_contrib_total_risk * 100

active_pct_factor_risk = active_factor_risk**2 / active_total_risk**2
active_pct_specific_risk = active_specific_risk**2 / active_total_risk**2
active_pct_total_risk = active_total_risk**2 / active_total_risk**2

active_pct_contrib_factor_risk = active_pct_factor_risk * active_contrib_factor_risk / np.sum(active_contrib_factor_risk)
active_pct_contrib_specific_risk = active_pct_specific_risk * active_contrib_specific_risk / np.sum(active_contrib_specific_risk)
active_pct_contrib_total_risk = active_pct_total_risk * active_contrib_total_risk / np.sum(active_contrib_total_risk)

# Security Factor Exposures
port_factor_exposure = np.dot(port.T, factor_exposures)
bench_factor_exposure = np.dot(bench.T, factor_exposures)
active_factor_exposure = np.dot(active.T, factor_exposures)

# Factor Risk Decomposition
port_factor_decomp_marginal = np.dot(factor_covar, port_factor_exposure.T) / port_factor_risk / 100
port_factor_decomp_contrib = port_factor_exposure.T * port_factor_decomp_marginal * 100
port_factor_decomp_pct_contrib = port_factor_decomp_contrib / port_factor_risk

bench_factor_decomp_marginal = np.dot(factor_covar, bench_factor_exposure.T) / bench_factor_risk / 100
bench_factor_decomp_contrib = bench_factor_exposure.T * bench_factor_decomp_marginal * 100
bench_factor_decomp_pct_contrib = bench_factor_decomp_contrib / bench_factor_risk

active_factor_decomp_marginal = np.dot(factor_covar, active_factor_exposure.T) / active_factor_risk / 100
active_factor_decomp_contrib = active_factor_exposure.T * active_factor_decomp_marginal * 100
active_factor_decomp_pct_contrib = active_factor_decomp_contrib / active_factor_risk

#==============================================================================
# The End
#==============================================================================