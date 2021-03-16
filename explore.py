import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats
from datetime import date 


def split_data(df):
    '''
    split our data,
    takes in a pandas dataframe
    returns: three pandas dataframes, train, test, and validate
    '''
    train_val, test = train_test_split(df, train_size=0.8, random_state=123)
    train, validate = train_test_split(train_val, train_size=0.7, random_state=123)
    
    
    return train, validate, test





def plot_variable_pairs(train, quant_vars):
    '''
    This function creates a pair plot of all of variables. We can see if there is correlation between them
    
    '''
    
    
    pair = sns.pairplot(data=train, vars=quant_vars, hue='fips', kind = 'reg', diag_kind="kde", markers=["o", "s", "D"], palette="Set2")
    plt.show()
     

def years_old(train):
    
    '''
    
    This function creates a new column to a pandas data frame
    
    '''
    
    today = date.today()

    train['age_of_home'] = today.year - train.yearbuilt
    
    return train


''' The following three functions will be put inside the plot_categorical_and_continuous_vars function '''

def plot_quant(train, quant_vars):
        for col in list(train.columns):
            
            if col in quant_vars:
                sns.boxplot(data = train, y = col)
                plt.show()

def plot_cat(train, cat_vars):
    for col in list(train.columns):
        if col in cat_vars:
            sns.distplot(train[col])
            plt.show()
    
def corr_map(train):
        
        sns.set(style="white")
        corr = train.corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        f, ax = plt.subplots(figsize=(12, 10))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.9, center=0, square=True, linewidths=.5, annot=True,cbar_kws={ "shrink": .5});

# ----------------------------------        
        
def plot_categorical_and_continuous_vars(train, target, cat_vars, quant_vars):
    
    '''
    This function will hold the two functions above that plot both categorical and quanitative variables
    '''
    
    plot_cat(train, cat_vars)
    
    plot_quant(train, quant_vars)
    
    corr_map(train)


        
        
    
    
    
    

        




    
        
    