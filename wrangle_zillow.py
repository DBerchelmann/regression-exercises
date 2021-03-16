  
import pandas as pd
import numpy as np
import os

# acquire
from env import host, user, password
from pydataset import data

# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split



os.path.isfile('telco_log_df.csv')


# Create helper function to get the necessary connection url.
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    
    

# Use the above helper function and a sql query in a single function.
def new_zillow_data():
    '''
    This function reads data from the Codeup db into a df.
    '''
    zillow_sql = "SELECT parcelid, propertylandusetypeid, propertylandusedesc, \
                 transactiondate, calculatedfinishedsquarefeet, bedroomcnt, \
                 bathroomcnt,fips, regionidzip, yearbuilt, taxvaluedollarcnt, \
                 assessmentyear, taxamount, latitude, longitude \
                 FROM predictions_2017 \
                 JOIN properties_2017 using (parcelid) \
                 JOIN propertylandusetype using (propertylandusetypeid) \
                 WHERE month(transactiondate) >= 05 and month(transactiondate) <= 06 and unitcnt = 1 or propertylandusetypeid = 261 or propertylandusetypeid = 264 \
                 or propertylandusetypeid = 273 or propertylandusetypeid = 276 \
                 ;" 
    
    
    return pd.read_sql(zillow_sql, get_connection('zillow'))

def clean_zillow(df):
    
    
    '''
    For this practice zillow data frame, we will be locating NaNs in different columns
    and removing those from the dataset.
    
    We will return: df, a cleaned pandas dataframe
    '''
    
    # Remove NaNs from finished square feet
    df.loc[df['calculatedfinishedsquarefeet'].isin(['NaN'])].head()
    indexsize = df.loc[df['calculatedfinishedsquarefeet'].isin(['NaN'])].index
    df.drop(indexsize , inplace=True)
    
    # Remove NaNs from zip code
    df.loc[df['regionidzip'].isin(['NaN'])].head()
    indexzip = df.loc[df['regionidzip'].isin(['NaN'])].index
    df.drop(indexzip , inplace=True)
    
    # Remove NaNs from tax amount
    df.loc[df['taxamount'].isin(['NaN'])].head()
    indextax = df.loc[df['taxamount'].isin(['NaN'])].index
    df.drop(indextax , inplace=True)
    
    

    
    return df

def split_data(df):
    '''
    split our data,
    takes in a pandas dataframe
    returns: three pandas dataframes, train, test, and validate
    '''
    train_val, test = train_test_split(df, train_size=0.8, random_state=123)
    train, validate = train_test_split(train_val, train_size=0.7, random_state=123)
    
    
    return train, validate, test
       
   

 # wrangle!
def wrangle_telco():
    '''
    wrangle_grades will read in our student grades as a pandas dataframe,
    clean the data
    split the data
    return: train, validate, test sets of pandas dataframes from telco_churn, stratified on total_charges
    '''
    df = clean_zillow(new_zillow_data())
    
    
    
    return split_data(df)