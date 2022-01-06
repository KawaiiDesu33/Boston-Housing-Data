#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 00:23:08 2022

@author: terenceau
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

path = '/Users/terenceau/Desktop/Python/Boston Housing Data/Boston.csv'

data = pd.read_csv(path)
header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX'	
           ,'PTRATIO','B', 'LSTAT','MEDV']

data.columns = header
          

### Correlation of Variables
data_corr = data.corr()
sns.heatmap(data_corr, cmap = 'viridis')


### Relationship Plots
sns.relplot(x = 'RM', y = 'MEDV', hue = 'CHAS', data = data, kind = 'scatter')
# Housing Price and Average Room Number - Positive Relationship

sns.relplot(x = 'TAX', y = 'MEDV', data = data, kind = 'scatter')
# Housing Price and Tax - No Clear Relationship - Break in Relationship

sns.relplot(x = 'AGE', y = 'MEDV', data = data, kind = 'scatter')
# Weak Negative Relationship - Increased Age of Property
# ONly for Low to Mid Ranged Property - Mid to High Property Retains Value

sns.relplot(x = 'CRIM', y = 'B', data = data, kind = 'scatter')
sns.relplot(x = 'CRIM', y = 'MEDV', data = data, kind = 'scatter')
# Low Crime Rate even with high proportion of Blacks
# No Strong Link Between Crime and House Prices - Only Few Data Points

sns.relplot(x = 'ZN', y = 'RM', data = data, kind = 'scatter')
sns.relplot(x = 'INDUS', y = 'RM', data = data, kind = 'scatter')
# Average Room in Housing does not matter depending on Residential Land Zone Area
# Weak Relationship - More Corporate Business - Lower Average Room Number


### Regressions

# - All Variables
reg_all_model = smf.ols(formula = 'MEDV ~ CRIM + ZN + INDUS + + CHAS + NOX + RM +\
                        AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT',
                        data = data)
reg_results = reg_all_model.fit()
print(reg_results.summary())
## Strong Fit, INDUS and AGE NOT Significant

# - Assigning Regression Values
reg_params = pd.DataFrame(reg_results.params)

int_coef = reg_params.loc['Intercept', 0]
CRIM_coef = reg_params.loc['CRIM', 0]
ZN_coef = reg_params.loc['ZN', 0]
INDUS_coef = reg_params.loc['INDUS', 0]
CHAS_coef = reg_params.loc['CHAS', 0]
NOX_coef = reg_params.loc['NOX', 0]
RM_coef = reg_params.loc['RM', 0]
AGE_coef = reg_params.loc['AGE', 0]
DIS_coef = reg_params.loc['DIS', 0]
RAD_coef = reg_params.loc['RAD', 0]
TAX_coef = reg_params.loc['TAX', 0]
PT_coef = reg_params.loc['PTRATIO', 0]
B_coef = reg_params.loc['B', 0]
LSTAT_coef = reg_params.loc['LSTAT', 0]
    
temp = []

# - Predicted Y hats
for i in range(0, 506, 1):
    a = (int_coef + CRIM_coef*data.iloc[i, 0] + ZN_coef*data.iloc[i, 1] +
        INDUS_coef*data.iloc[i, 2] + CHAS_coef*data.iloc[i, 3]
        + NOX_coef*data.iloc[i, 4] + RM_coef*data.iloc[i, 5] + AGE_coef*data.iloc[i, 6]
        + DIS_coef*data.iloc[i, 7] + RAD_coef*data.iloc[i, 8] + 
        TAX_coef*data.iloc[i, 9] + PT_coef*data.iloc[i, 10] + 
        B_coef*data.iloc[i, 11] + LSTAT_coef*data.iloc[i, 12])
    temp.append(a)
    
yhat = pd.DataFrame(temp)
yhat.columns = ['yhat']

data_tot = data.join(yhat)


# Plots of reg_results
figure1 = sm.graphics.plot_regress_exog(reg_results, 'CRIM')

figure2 = sm.graphics.plot_regress_exog(reg_results, 'RM')

figure_resid = sm.graphics.plot_leverage_resid2(reg_results)

figure3 = sm.graphics.qqplot(data['MEDV'], fit = True, line = '45')
plt.show()

# Plotting Residuals
resid = data_tot['MEDV'] - data_tot['yhat']
resid = pd.DataFrame(resid)
resid.columns = ['residual']

resid_mean = resid.mean()
resid_var = resid.var()

norm_resid = (resid-resid_mean)/resid_var
norm_resid = pd.DataFrame(norm_resid)
norm_resid.columns = ['norm_resid']

data_tot = data_tot.join(resid)
data_tot = data_tot.join(norm_resid)

x = range(0, 506, 1)

plt.scatter(x = x, y = data_tot['norm_resid'])
plt.ylabel('Normalised Residuals')
plt.show()
