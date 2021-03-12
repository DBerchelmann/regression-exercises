import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.preprocessing import QuantileTransformer
from wrangle import split_data, wrangle_telco


   
def get_scaled():
    
    train, validate, test = wrangle_telco()
    
    scaler = sklearn.preprocessing.RobustScaler()
    
    scaler.fit(train)
    
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)
    
    # turn the numpy arrays into dataframes
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)
    
 
    
    return train_scaled, validate_scaled, test_scaled