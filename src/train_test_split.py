# #!/usr/bin/env python

# Functions to create the index based on the training data and then to map 
# the test index to the closest matching training index (i.e. threshold).


import bisect

import numpy as np
import pandas as pd


def index_train_data(train_files, column_names):
    '''
    Creates a common index of weights based on the training data. 

    Takes a dataset of filtrations, with each filtration stored as
    a pandas DataFrame, and builds a common index to standardize the
    data. Each filtration gets reindexed with all the thresholds in the
    entire training dataset, and the values of graph descriptor function
    are forward-filled if an individual filtration did not have
    a specific value.

    Parameters
    ----------
    train_files : list
        A list of pd.DataFrames where the edge weight is the index of
        the dataframe, and the columns are the node label histograms 
    column_names: list 
        A list of column names from the pd.DataFrames in train_files

    Returns
    -------
    X : list
        A list of the reindexed filtrations.
    df_index: index
        The union of all edge weights included in the training data.

    '''
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

    return X, df_index


def index_test_data(test_files, column_names, train_index):
    '''
    Reindexes the test data filtrations with the training data edge
    weights.
    
    Takes the test data filtrations and reindexes them with the training
    data edge weights. If a given test edge weight does not exist in the
    training dataset, it is replaced with the closets edge weight in the
    training data.
    
    Parameters
    ----------
    test_files: list
        A list of pd.DataFrames where the edge weight is the index of
        the dataframe, and the columns are the node label histograms 
    column_names: list 
        A list of column names from the pd.DataFrames in test_files
    train_index: index
        The edge weights in the training data

    Returns
    -------
    X : list
        The test dataset that has been reindexed to the training
        thresholds. It is a list of pd.DataFrames, one per data point.

    '''
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

    return X
