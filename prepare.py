import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.preprocessing import QuantileTransformer
from wrangle import split_data, wrangle_telco


   
def get_scaled():
    
    '''
    This function is used to scale data using the min/max scale function.
    Would need to update if wanting to use standard, robust, quantile or other type.
    
    '''
    
    # call data in from the wrangle function
    
    train, validate, test = wrangle_telco()
    
    # make our scaler below
    
    scaler = sklearn.preprocessing.RobustScaler()
    
    # be sure to fit the data to the TRAIN dataset, nothing else
    
    scaler.fit(train)
    
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)
    
    # turn the numpy arrays into dataframes
    
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)
    
 
    
    return train_scaled, validate_scaled, test_scaled