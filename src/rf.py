''' Random forest classifier '''

import csv
import glob
import os
import argparse
import warnings 
import time
from tqdm import tqdm
import random
import json

import pandas as pd
from scipy.spatial import distance

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from utils import *
from train_test_split import *


def run_rf(X, y, n_iterations=10):
    random.seed(42)

    iteration_metrics = create_metric_dict()
    iteration_accuracies = []
    for iteration in range(n_iterations):
        
        fold_metrics = create_metric_dict()
        fold_accuracies = []    
                
        cv = StratifiedKFold(
            n_splits=10,
            random_state=42 + iteration,
            shuffle=True
            ) 

        for train_index, test_index in cv.split(np.zeros(len(y)), y):
            X_train = [X[i] for i in train_index]
            y_train = [y[i] for i in train_index]
            
            X_test = [X[i] for i in test_index]
            y_test = [y[i] for i in test_index]

            clf = RandomForestClassifier(
                    max_depth=None, 
                    random_state=0,
                    n_estimators=1000, 
                    class_weight="balanced",
                    ) 

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            with warnings.catch_warnings(): 
                warnings.filterwarnings('ignore')

                fold_metrics = compute_fold_metrics(
                        y_test, 
                        y_pred,
                        fold_metrics
                        )
        iteration_metrics = update_iteration_metrics(
                fold_metrics,
                iteration_metrics
                )
    
    
    print_iteration_metrics(iteration_metrics)


def run_rf_inductive(original_X, y, column_names, n_iterations=10):
    random.seed(42)

    n_graphs = len(y)
    iteration_metrics = create_metric_dict()
    iteration_accuracies = []
    for iteration in range(n_iterations):
        print(iteration) 
        fold_metrics = create_metric_dict()
        fold_accuracies = []    
                
        cv = StratifiedKFold(
            n_splits=10,
            random_state=42 + iteration,
            shuffle=True
            ) 

        for train_index, test_index in cv.split(np.zeros(len(y)), y):

            train_files = [original_X[i] for i in train_index]
            test_files = [original_X[i] for i in test_index]

            X1, index = index_train_data(
                    train_files=train_files, 
                    column_names=column_names)
            X2 = index_test_data(
                    test_files=test_files, 
                    column_names=column_names,
                    train_index=index)       
    
            X = [[[0] for i in range(n_graphs)] for j in range(len(column_names))]

            n_index = len(index)
            selected_thresholds =  None # [None, "change_points", "average_thresholds"]
            X = [[[0] for i in range(n_graphs)] for j in range(len(column_names))]
            
            for i, col in enumerate(column_names):
                for idx, graph in enumerate(train_index):
                    X[i][graph] = X1[i][idx]
                    
                for idx, graph in enumerate(test_index):                   
                    X[i][graph] = X2[i][idx]
           
            n_node_labels = len(X)
            new_X = []

            for graph_idx in range(n_graphs):
                graph_representation = []
                for node_label_idx in range(n_node_labels):
                    graph_representation.extend(X[node_label_idx][graph_idx])
                new_X.append(graph_representation)
            
            X_train = [new_X[i] for i in train_index]
            y_train = [y[i] for i in train_index]
            
            X_test = [new_X[i] for i in test_index]
            y_test = [y[i] for i in test_index]

            clf = RandomForestClassifier(
                    max_depth=None, 
                    random_state=0,
                    n_estimators=1000, 
                    class_weight="balanced",
                    ) 

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            with warnings.catch_warnings(): 
                warnings.filterwarnings('ignore')

                fold_metrics = compute_fold_metrics(
                        y_test, 
                        y_pred,
                        fold_metrics
                        )

        iteration_metrics = update_iteration_metrics(
                fold_metrics,
                iteration_metrics
                )
    
    
    print_iteration_metrics(iteration_metrics)

