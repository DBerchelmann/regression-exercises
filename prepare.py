import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.preprocessing import QuantileTransformer
from wrangle import split_data


   
def get_scaled():
    
    train, valid, test = split_data(df)
    
    scaler = sklearn.preprocessing.RobustScaler()
    
    scaler.fit(train)
    
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)
    
    
    
 
    
    return train_scaled, validate_scaled, test_scaled