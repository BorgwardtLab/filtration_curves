# #!/usr/bin/env python
#
# Provides a simple way to perform classification based on the
# curves arising from the calculated filtrations that uses a 
# train test split.

import argparse
import warnings
import random
import csv
import time
from bisect import bisect_left
import glob
from tqdm import tqdm
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig

from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

from select_thresholds import *
from train_test_split import *
from demo_filtration_baseline import get_topological_feature 
from filtration import *


def normalize(matrix):
    '''
    Normalizes a kernel matrix by dividing through the square root
    product of the corresponding diagonal entries. This is *not* a
    linear operation, so it should be treated as a hyperparameter.
    :param matrix: Matrix to normalize
    :return: Normalized matrix
    '''

    # Ensures that only non-zero entries will be subjected to the
    # normalisation procedure. The remaining entries will be kept
    # at zero. This prevents 'NaN' values from cropping up.
    mask = np.diagonal(matrix) != 0
    n = len(np.diagonal(matrix))
    k = np.zeros((n, ))
    k[mask] = 1.0 / np.sqrt(np.diagonal(matrix)[mask])

    return np.multiply(matrix, np.outer(k, k))

if __name__ == '__main__':


    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', help='Input file(s)', nargs='+')
     
    args = parser.parse_args()
    
    # stores the graphs and graph labels
    y = []
    list_of_df = []

    files = sorted(args.FILES)
    dataset = args.FILES[0].split("/")[2]
    n_graphs = len(files)
    n_iterations = 10

    # create list of dataframes (i.e. data) and y labels
    for idx, filename in enumerate(files):
        df = pd.read_csv(filename, header=0, index_col='weight')
        df = df.loc[~df.index.duplicated(keep="last")] 
        y.append(df['graph_label'].values[0])
        df = df.drop(columns='graph_label')
        list_of_df.append(df)
    y = LabelEncoder().fit_transform(y)

    # get the column names
    example_file = list(pd.read_csv(args.FILES[0], header=0, index_col="weight").columns)
    column_names = [c for c in example_file if c not in ["graph_label"]]

    # run repeated k-fold cross validation
    iteration_accuracies = []
        
    for iteration in range(n_iterations):
        start_time = time.time()
        fold_accuracies = []
        
        cv = StratifiedKFold(
                n_splits=10,
                random_state=42 + iteration,
                shuffle=True
            ) 

        for train_index, test_index in cv.split(np.zeros(n_graphs), y):
            train_files = [list_of_df[i] for i in train_index]
            test_files = [list_of_df[i] for i in test_index]
            
            # build training index and reindex the test files to it
            X1, index = index_train_data(
                    train_files=train_files, 
                    column_names=column_names)
            X2 = index_test_data(
                    test_files=test_files, 
                    column_names=column_names,
                    train_index=index)        

            n_index = len(index)
            selected_thresholds =  None # [None, "change_points", "average_thresholds"]
            X = [[[0] for i in range(n_graphs)] for j in range(len(column_names))]
            
            for i, col in enumerate(column_names):
                for idx, graph in enumerate(train_index):
                    X[i][graph] = X1[i][idx]
                    
                for idx, graph in enumerate(test_index):                   
                    X[i][graph] = X2[i][idx]
            
            # Make a single vector
            # calculate the pairwise distance of all pairs of graphs
            D = np.zeros((n_graphs, n_graphs))
            all_X = [[] for i in range(n_graphs)]
            for i, col in enumerate(column_names):
                
                X_i = np.array(X[i])
                max_i = np.max(X_i)
                if max_i == 0:
                    max_i = 1
                X_i /= max_i    
                for j in range(n_graphs):
                    all_X[j].extend(X_i[j])

                if selected_thresholds is not None:
                    sample_indices = return_selected_thresholds(
                            method=selected_thresholds, 
                            files=train_files,
                            ind=index,
                            weights=None)
                    X_i = [X_i[j][sample_indices] for j in range(n_graphs)]

                #D += -pairwise_distances(X_i, metric="l2")
                #D += rbf_kernel(X_i, gamma=10)
                #D += linear_kernel(X_i)
            #D = rbf_kernel(all_X, gamma=0.001)
            
            # Separate gram matrix into training and test
            #D_train = D[train_index][:, train_index]
            D_train = [all_X[i] for i in train_index]
            y_train = y[train_index]
            
            #D_test = D[test_index][:, train_index]
            D_test = [all_X[i] for i in test_index]
            y_test = y[test_index]

            grid_search = RandomForestClassifier(
                    max_depth=None, 
                    random_state=0,
                    class_weight="balanced", 
                    n_estimators=1000) 
            #clf = SVC(
            #    kernel='rbf',
            #    max_iter=500,
            #    class_weight='balanced'
            #)

            #param_grid = {
            #    'C': 10. ** np.arange(-4, 3),
            #    'gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
            #}

            #grid_search = GridSearchCV(
            #    clf,
            #    param_grid,
            #    cv=3,
            #)

            with warnings.catch_warnings(): 
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                grid_search.fit(D_train, y_train)

            y_pred = grid_search.predict(D_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(accuracy)

            fold_accuracies.append(accuracy)

            #print(f'{accuracy * 100:2.2f}') # fold accuracy
        end_time = time.time()
        iteration_accuracies.append(np.mean(fold_accuracies))
        print(f'Iteration {iteration}: {np.mean(fold_accuracies) * 100:2.2f}') # repeated 10fold accuracy for a single iteration
        print(end_time - start_time)

    mean_accuracy = np.mean(iteration_accuracies) * 100
    sdev_accuracy = np.std(iteration_accuracies) * 100
    print(f'{mean_accuracy:2.2f} +- {sdev_accuracy:2.2f}') #mean across iterations

    end = time.time()
    #print(end-start)
