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
# Portfolio, benchmark, and active risks
#==============================================================================

# Class that performs all risk model calculations
class RiskModel():
    
    def __init__(self, port):
        self.port = port
    
    # Total risk (portfolio-level)
    def factor_risk_total(self):
        return float(np.sqrt(np.dot(np.dot(self.port.T, factor_risk), self.port)))
    def specific_risk_total(self):
        return float(np.sqrt(np.dot(np.dot(self.port.T, specific_risk), self.port)))
    def total_risk_total(self):
        return float(np.sqrt(np.dot(np.dot(self.port.T, total_risk), self.port)))

    # Marginal risk (stock-level)
    def factor_risk_marginal(self):
        return np.concatenate(np.dot(factor_risk, self.port) / self.factor_risk_total() / 100, axis=0)
    def specific_risk_marginal(self):
        return np.concatenate(np.dot(specific_risk, self.port) / self.specific_risk_total() / 100, axis=0)
    def total_risk_marginal(self):
        return np.concatenate(np.dot(total_risk, self.port) / self.total_risk_total() / 100, axis=0)
    
    # Contribution to risk (stock-level)
    def factor_risk_contrib(self):
        return np.concatenate(np.array(self.port), axis=0) * self.factor_risk_marginal() * 100
    def specific_risk_contrib(self):
        return np.concatenate(np.array(self.port), axis=0) * self.specific_risk_marginal() * 100
    def total_risk_contrib(self):
        return np.concatenate(np.array(self.port), axis=0) * self.total_risk_marginal() * 100
    
    # Percent contribution to risk (stock-level)
    def factor_risk_pct_contrib(self):
        return ((self.factor_risk_total()**2 / self.total_risk_total()**2) * 
                self.factor_risk_contrib() / np.sum(self.factor_risk_contrib()))
    def specific_risk_pct_contrib(self):
        return ((self.specific_risk_total()**2 / self.total_risk_total()**2) * 
                self.specific_risk_contrib() / np.sum(self.specific_risk_contrib()))
    def total_risk_pct_contrib(self):
        return ((self.total_risk_total()**2 / self.total_risk_total()**2) * 
                self.total_risk_contrib() / np.sum(self.total_risk_contrib()))

    # Portfolio factor risk decomposition
    def factor_decomp_marginal(self):
        return np.dot(factor_covar, np.dot(self.port.T, factor_exposure).T) / self.factor_risk_total() / 100
    def factor_decomp_contrib(self):
        return np.dot(self.port.T, factor_exposure).T * self.factor_decomp_marginal() * 100
    def factor_decomp_pct_contrib(self):
        return self.factor_decomp_contrib() / self.factor_risk_total()

#==============================================================================
# Data import
#==============================================================================

portfolio = pd.read_csv(PORT_FILE, index_col='TICKER', header=0) # Portfolio
benchmark = pd.read_csv(BENCH_FILE, index_col='TICKER', header=0) # Benchmark
active = portfolio - benchmark # Active portfolio

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
# mntm: 1. Factor: Momentum
# ====  2. Definition: Price change from 12 months ago to 1 month ago 
#       3. Scaling: The values are normalized

factor_size = np.log(pri*shr)
factor_size_norm = factor_size.sub(factor_size.mean(axis=1), axis=0).div(factor_size.std(axis=1), axis=0)

factor_valu  = eps.shift(2).rolling(window=12, min_periods=None).sum() / pri
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

betas = [] # Betas
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

# Risk matrices for factor, specific, and total risk (of size NxN)
factor_risk = np.dot(np.dot(factor_exposure, factor_covar), factor_exposure.T)
specific_risk = np.diag((np.std(resid, ddof=1, axis=0) * np.sqrt(12))**2)
total_risk = factor_risk + specific_risk

#==============================================================================
# The End
#==============================================================================