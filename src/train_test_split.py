# #!/usr/bin/env python

# Functions to create the index based on the training data and then to map 
# the test index to the closest matching training index (i.e. threshold).


import bisect

import numpy as np
import pandas as pd


def index_train_data(train_files, column_names):
    '''Creates the threshold index based on the training data'''
    X = [[] for i in column_names]
    
    # Create shared index of all training thresholds
    df_index = None
    for df in train_files: 
                
        if df_index is None:
            df_index = df.index
        else:
            df_index = df.index.union(df_index)
            df_index = df_index[~df_index.duplicated()] 

    # reindex with the full training thresholds
    for df in train_files: 
        tmp_df = df.reindex(df_index)           # create missing values
        tmp_df = tmp_df.fillna(method='ffill')  # forward-filling for consistency
        tmp_df = tmp_df.fillna(0)               # replace values at the beginning  

        for i, col in enumerate(column_names):
            X[i].append(tmp_df.iloc[:,i].transpose().to_numpy())

    return(X, df_index)


def index_test_data(test_files, column_names, train_index):
    '''Reindexes the test datasets using the closest threshold from the training data'''

    X = [[] for i in column_names]   
    
    for df in test_files:
        test_index = df.index.tolist()                  # thresholds
        test_values = [df.loc[i] for i in test_index]   # histogram counts

        # map test threshold to closest training threshold
        tr_index = [bisect.bisect_left(train_index, k) for k in test_index] # training threshold index
        reindex_to_train = [train_index[min(i, len(train_index)-1)] for i in tr_index] # training threshold weight
        
        # reindex df and populate with the test values
        tmp_df = df.reindex(train_index) 
        for idx, value in enumerate(reindex_to_train):
            tmp_df.loc[value] = test_values[idx]
        tmp_df = tmp_df.fillna(method='ffill')  # forward-filling for consistency
        tmp_df = tmp_df.fillna(0)               # replace values at the beginning 

        for i, col in enumerate(column_names):
            X[i].append(tmp_df.iloc[:,i].transpose().to_numpy())

    return(X)
