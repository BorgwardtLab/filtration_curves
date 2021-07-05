
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
import itertools

import numpy as np
import pandas as pd
import igraph as ig

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

#from select_thresholds import *
from train_test_split import *


#def label_distribution_graph(graph, label_to_index):
#    '''
#    Calculates the node label distribution of a filtration, using a map
#    that stores index assignments for labels.
#
#    :param filtration: A filtration of graphs
#    :param label_to_index: A map between labels and indices, required to
#    calculate the histogram.
#
#    :return: Label distributions along the filtration. Each entry is
#    a tuple consisting of the weight of the filtration followed by a
#    count vector.
#    '''
#
#    # Will contain the distributions as count vectors; this is
#    # calculated for every step of the filtration.
#
#    labels = graph.vs['label']
#    counts = np.zeros(len(label_to_index))
#
#    for label in labels:
#        index = label_to_index[label]
#        counts[index] += 1
#
#        # The conversion ensures that we can serialise everything later
#        # on into a `pd.series`.
#
#    return counts 
#
#
#def graphs_to_label_counts(graphs):
#
#    node_labels = sorted(set(
#        itertools.chain.from_iterable(graph.vs['label'] for graph in graphs)
#    ))
#
#    label_to_index = {
#        label: index for index, label in enumerate(sorted(node_labels))
#    }
#
#
#    counts = []
#    for graph in graphs:
#            
#        distributions = label_distribution_graph(graph, label_to_index)
#        counts.append(distributions)
#
#    return(counts)



if __name__ == '__main__':


    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Input file(s)')
    parser.add_argument('--n', type=int, default=100)

    args = parser.parse_args()
    
    # stores the graphs and graph labels
    y = []
    list_of_df = []
    files = sorted(glob.glob(os.path.join(
        "../preprocessed_data/labeled_datasets/" + args.dataset + "/",
        '*.csv'
        )))
    dataset = args.dataset.split("_nodes")[0]
    n_graphs = len(files)
    n_iterations = 10

    # create list of dataframes (i.e. data) and y labels
    for idx, filename in enumerate(tqdm(files)):
        df = pd.read_csv(filename, header=0, index_col='weight')
        df = df.loc[~df.index.duplicated(keep="last")] 
        y.append(df['graph_label'].values[0])
        df = df.drop(columns='graph_label')
        list_of_df.append(df)
    y = LabelEncoder().fit_transform(y)
    
    # get the column names
    example_file = list(pd.read_csv(files[0], header=0, index_col="weight").columns)
    column_names = [c for c in example_file if c not in ["graph_label"]]
    #
    # reindex
    list_of_df, index = index_train_data(list_of_df, column_names)

    all_X = []
    for idx, sample in enumerate(list_of_df[0]):
        person = []
        for j in range(len(list_of_df)):
            person.extend(list_of_df[j][idx])
        all_X.append(person)

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
            train_files = [all_X[i] for i in train_index]
            test_files = [all_X[i] for i in test_index]
            
            D_train = [all_X[i] for i in train_index]   
            y_train = y[train_index]

            D_test = [all_X[i] for i in test_index]
            y_test = y[test_index]

            clf = RandomForestClassifier(
                    max_depth=None, 
                    random_state=0,
                    class_weight="balanced", 
                    n_estimators=1000) 

            with warnings.catch_warnings(): 
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                clf.fit(D_train, y_train)

            y_pred = clf.predict(D_test)
            accuracy = accuracy_score(y_test, y_pred)

            fold_accuracies.append(accuracy)

            print(f'{accuracy * 100:2.2f}') # fold accuracy
        end_time = time.time()
        iteration_accuracies.append(np.mean(fold_accuracies))
        print(iteration_accuracies)

    mean_accuracy = np.mean(iteration_accuracies) * 100
    sdev_accuracy = np.std(iteration_accuracies) * 100
    print(f'{mean_accuracy:2.2f} +- {sdev_accuracy:2.2f}') #mean across iterations

    end = time.time()
    #print(dataset, end-start)
