import pandas as pd
import numpy as np
import statsmodels.api as sm
import sqlalchemy as sq

import os


def import_data():
    '''
    INPUT: string of filepath to csv data filepath
    OUTPUT: pandas dataframe

    Use this function to read in the SBA data. I hardcoded the approval date to a
    datetime object, so this function will only work on the SBA dataframe.
    '''
    key = os.getenv('SBA_DWH')
    engine = sq.create_engine(key)
    return engine


def feature_engineer(df):
    #Do stuff here
    return df


def dummify_variables(df, var_names):
    '''
    INPUT: pandas df from import_data fxn
           list of column names to dummify
    OUTPUT: pandas df with dummified variables from list
    '''
    if 'grade' in var_names:
        grade_dummies = pd.get_dummies(
            pd.cut(df['grade'], bins=[0, 4, 7, 10, 13]))
        grade_dummies.columns = ['low_grade',
                                 'mid_grade', 'high_grade', 'higher_grade']

    return df.join(grade_dummies).drop(['grade', 'low_grade'], axis=1)


def train_model(data, dropped_columns, print_summary=False):
    y = data['Success']
    X = data.drop(dropped_columns, axis=1)
    X = sm.add_constant(X)
    model = sm.Logit(y, X)
    trained_model = model.fit()
    if print_summary:
        print trained_model.summary()
    else:
        return trained_model

if __name__ == '__main__':
    #dropped_columns=['Unnamed: 0', 'Column', 'BorrName', 'BorrStreet', 'BorrCity', 'BorrState',\
    # 'BankStreet', 'BankCity', 'BankState', 'SBADistrictOffice', 'ProjectState', 'ProjectCounty',\
    #  'ThirdPartyLender_State', 'ThirdPartyLender_Name', 'ThirdPartyLender_City', 'ChargeOffDate']
    engine = import_data()
    print engine.table_names()
    with engine.begin() as conn:
        df = pd.read_sql_table('sba_sfdo', conn, schema='stg_analytics')
    print len(df)
    #print df.describe()
    # train_model(SBA_data, print_summary=True)
