import pandas as pd
import numpy as np
import statsmodels.api as sm
import sqlalchemy as sq

import os


def import_data():
    '''
    OUTPUT: pandas dataframe

    Use this function to read in the SBA data. If running on your computer, you need to change the key
    to your own enviornment variable that has the database's address
    '''
    key = os.getenv('SBA_DWH')
    engine = sq.create_engine(key)
    with engine.begin() as conn:
        df = pd.read_sql_table('sba_sfdo', conn, schema='stg_analytics')
    return df


def feature_engineer(df):
    # Do stuff here
    return engineered_df


def dummify_variables(df):
    '''
    INPUT: pandas df from import_data fxn
    OUTPUT: pandas df with dummified variables from list

    Feel free to add more variables to dummify
    '''
    df['is_7a_loan'] = pd.get_dummies(df['program'],drop_first=True)
    df.drop('program', axis=1, inplace=True)
    return dummified_df


def create_x_y(df, dropped_columns):
    '''
    INPUT: pandads dataframe from dummify_variables fxn,
    list of strings that are column names of columns to drop
    OUTPUT: feature matrix X and response vector y
    '''
    y = df['Success']
    df.drop(df['Success'], axis=1, inplace=True)
    X = df.drop(dropped_columns, axis=1)
    return X, y


def train_model(X, y, dropped_columns, print_summary=False):
    '''
    INPUT: pandas dataframes containing the target column y and the feature matrix X,
    list of strings containing column names that weren't dropped earlier in the pipeline,
    and optional boolean argument to print out the model summary
    OUTPUT: fitted logistic regression model

    This function takes in the X and y dataframes created from fxn create_x_y
    '''
    X = sm.add_constant(X)
    model = sm.Logit(y, X)
    fitted_model = model.fit()
    if print_summary:
        print fitted_model.summary()
    return fitted_model

if __name__ == '__main__':
    dropped_columns=['borr_name', 'borr_street', 'borr_city', 'borr_state',\
     'bank_street', 'bank_city', 'bank_state', 'sba_district_office', 'project_state',\
     'third_party_lender_state', 'cdc_street', 'cdc_state']

    original_SBA_data = import_data()

    dummified_df = dummify_variables(df)

    X, y = create_x_y(dummified_df, dropped_columns)

    train_model(SBA_data, print_summary=True)
