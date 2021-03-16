import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from math import sqrt
import seaborn as sns

from statsmodels.formula.api import ols
from math import sqrt


import warnings
warnings.filterwarnings('ignore')


def plot_residuals(x, y, df):
    
    
    
    df['residual'] = y - df.yhat
    df['baseline_residual'] = y - df.baseline
    
    
    
    fig, ax = plt.subplots(2, figsize=(10, 8))

    ax[0].scatter(x, df.residual)
    ax[0].axhline(y = 0, ls = ':')
    ax[0].set_title('OLS model residuals');


    ax[1].scatter(x, df.baseline_residual)
    ax[1].axhline(y = 0, ls = ':')
    ax[1].set_title('Baseline Residuals');

    
def regression_errors(x, y, df):
    
    df['residual'] = y - df.yhat
    df['baseline_residual'] = y - df.baseline
    
    df['residual^2'] = df.residual**2
    df['baseline_residual^2'] = df.baseline_residual**2
    
    SSE = df['residual^2'].sum()
    SSE_baseline = df['baseline_residual^2'].sum()
    
    print(f'The SSE Baseline {round(SSE_baseline, 2)} & and SSE is {round(SSE, 2)}')
    
    print('\n')
    
    TSS = SSE_baseline
    
    print(f'The total sum of squares is {round(TSS, 2)}')
      
    print('\n')
    
    ESS = TSS - SSE
    
    print(f'The explained sum of squares is {round(ESS, 2)}')
    
    print('\n')
    
    MSE = SSE/len(df)
    MSE_baseline = SSE_baseline/len(df)
    
    print(f'The mean squared error is {round(MSE, 2)} & the MSE baseline is {round(MSE_baseline, 2)}')
    
    print('\n')
    
    RMSE = sqrt(MSE)
    RMSE_baseline =  sqrt(MSE_baseline)
    
    print(f'The root mean squared error is {round(RMSE, 2)} & the RMSE baseline is {round(RMSE_baseline, 2)}')
    
    print('\n')
    df_eval = pd.DataFrame(np.array(['SSE','MSE','RMSE']), columns=['metric'])
    df_baseline_eval = pd.DataFrame(np.array(['SSE_baseline','MSE_baseline','RMSE_baseline']), columns=['metric'])

    df_eval['model_error'] = np.array([SSE, MSE, RMSE])
    df_baseline_eval['model_error'] = np.array([SSE_baseline, MSE_baseline, RMSE_baseline])

    print(df_eval)
    print(df_baseline_eval)


def baseline_mean_errors(x, y, df):
    
    df['residual'] = y - df.yhat
    df['baseline_residual'] = y - df.baseline
    
    df['residual^2'] = df.residual**2
    df['baseline_residual^2'] = df.baseline_residual**2
    
    SSE2 = mean_squared_error(y, df.yhat)*len(df)
    SSE2_baseline = mean_squared_error(y, df.baseline)*len(df)
    
    print(f'the SSE2 is {round(SSE2, 2)} & the baseline is {round(SSE2_baseline, 2)} .')
    
    print('\n')
    
    MSE2 = mean_squared_error(y, df.yhat)
    MSE2_baseline = mean_squared_error(y, df.baseline)
    
    print(f'the MSE2 is {round(MSE2, 2)} & the baseline is {round(MSE2_baseline, 2)} .')
    
    print('\n')
    
    RMSE2 = sqrt(mean_squared_error(y, df.yhat))
    RMSE2_baseline = sqrt(mean_squared_error(y, df.baseline))
    
    print(f'the RMSE2 is {round(RMSE2, 2)} & the baseline is {round(RMSE2_baseline, 2)}  .')
    
    print('\n')
    
      
    df_eval = pd.DataFrame(np.array(['SSE2','MSE2','RMSE2']), columns=['metric'])
    df_baseline_eval = pd.DataFrame(np.array(['SSE2_baseline','MSE2_baseline','RMSE2_baseline']), columns=['metric'])

    df_eval['model_error'] = np.array([SSE2, MSE2, RMSE2])
    df_baseline_eval['model_error'] = np.array([SSE2_baseline, MSE2_baseline, RMSE2_baseline])

    print(df_eval)
    print(df_baseline_eval)

    


def better_than_baseline(x, y, df):
    
    df['residual'] = y - df.yhat
    df['baseline_residual'] = y - df.baseline
    
    df['residual^2'] = df.residual**2
    df['baseline_residual^2'] = df.baseline_residual**2
    
    SSE = df['residual^2'].sum()
    SSE_baseline = df['baseline_residual^2'].sum()
    
    if SSE < SSE_baseline:
        return (f'The sum of squared errors model of {round(SSE, 2)} performs better than the residual result of {round(SSE_baseline, 2)}')
    
    else:
        return (f'The sum of squared errors residual {round(SSE_baseline, 2)} performs better than the SSE model of {round(SSE, 2)}')
    
    
def model_significance(x, y, f):
    
    


    
    