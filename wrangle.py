  
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
def new_telco_data():
    '''
    This function reads data from the Codeup db into a df.
    '''
    sql_query = 'SELECT customer_id, monthly_charges, tenure, total_charges \
                FROM customers \
                JOIN contract_types USING (contract_type_id) \
                WHERE contract_type = "Two year" ;'
    
    
    return pd.read_sql(sql_query, get_connection('telco_churn'))

# Let's translate our work into reproducable functions

def acquire_telco(cached=False):


    '''
  This funtion is pulling in the new_telco_data function and createing a .csv called 
  telco_log_df, if not already made.
    '''
        


    if cached == False or os.path.isfile('telco__log_df.csv') == False:
        
        
        # Read fresh data from db into a DataFrame.
        df = new_telco_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('telco_log_df.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('telco_log_df.csv', index_col=0)
        
 
    return df

def clean_telco(df):
    
    
    '''
    Takes in a df of student exam grades and cleans the data appropriately by dropping null values,
    replacing whitespace,
    and converting data to numerical data types
    as well as dropping student_id column from the dataframe
    
    return: df, a cleaned pandas dataframe
    '''
    
    
    df["total_charges"] = pd.to_numeric(df.total_charges, errors='coerce')
    
    df.total_charges = df.total_charges.replace(r'^\s*$', np.nan, regex=True)
    
    df = df.fillna(0)
    
    
    df = df.dropna()
    
    df.index = df.customer_id
    df = df.drop(columns='customer_id')
    
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
    df = clean_telco(acquire_telco())
    
    
    
    return split_data(df)