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
reg_all_model = smf.ols(formula = 'MEDV ~ CRIM + ZN + INDUS + NOX + RM +\
                        AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT',
                        data = data)
reg_results = reg_all_model.fit()
print(reg_results.summary())

## Strong Fit, INDUS and AGE NOT Significant

# Plots of reg_results
figure1 = sm.graphics.plot_regress_exog(reg_results, 'CRIM')

figure2 = sm.graphics.plot_regress_exog(reg_results, 'RM')

figure_resid = sm.graphics.plot_leverage_resid2(reg_results)







