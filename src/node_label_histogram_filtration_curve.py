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
from utils import *
from train_test_split import *
from rf import *
from preprocessing.filtration import filtration_by_edge_attribute


def create_curve(
        source_path="../data/labeled_datasets/BZR_MD",
        output_path="../data/labeled_datasets/preprocessed_data/BZR_MD/"
        ):
    ''' Creates the filtration curves and saves each curve as a csv in
    an output directory for easier handling in the future '''

    # get all file names
    filenames = sorted(
        glob.glob(os.path.join(source_path, '*.pickle'))
        )

    # load all graphs
    graphs = [
        ig.read(filename, format='picklez') for filename in tqdm(filenames)
    ]

    # sometimes the edge weight is stored as an edge attribute; we will
    # change this to be an edge weight
    for graph in graphs:
        graph.es['weight'] = [e['attribute'] for e in graph.es]

    # Get all potential node labels to make sure that the distribution
    # can be calculated correctly later on.
    node_labels = sorted(set(
        itertools.chain.from_iterable(graph.vs['label'] for graph in graphs)
    ))

    label_to_index = {
        label: index for index, label in enumerate(sorted(node_labels))
    }
   
    # build the filtration using the edge weights
    filtrated_graphs = [
        filtration_by_edge_attribute(
            graph,
            attribute='weight',
            delete_nodes=True,
            stop_early=True
        )
        for graph in tqdm(graphs)
    ]
    
    # Create a data frame for every graph and store it; the output is
    # determined by the input filename, albeit with a new extension.
    for index, filtrated_graph in enumerate(tqdm(filtrated_graphs)):
        
        df = pd.DataFrame(columns=['graph_label, weight'].extend(node_labels))

        distributions = node_label_distribution(filtrated_graph, label_to_index)

        for weight, counts in distributions:

            row = {
                'graph_label': graphs[index]['label'],
                'weight': weight
            }

            row.update({
                str(node_label): count for node_label, count in
                zip(node_labels, counts)
            })

            df = df.append(row, ignore_index=True)
        
        output_name = os.path.basename(filenames[index])
        output_name = os.path.splitext(output_name)[0] + '.csv'
        output_name = os.path.join(output_path, output_name)

        os.makedirs(output_path, exist_ok=True)

        df.to_csv(output_name, index=False)


def load_curves(args):
    ''' Load the stored curves '''
    
    # stores the graphs and graph labels
    files = sorted(glob.glob(os.path.join(
        "../data/labeled_datasets/preprocessed_data/" + args.dataset + "/", '*.csv'
        )))

    y = []
    list_of_df = []

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

    return list_of_df, y, column_names


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--dataset', 
            help='dataset'
            )
    parser.add_argument(
            '--method',
            default="transductive",
            type=str,
            help="transductive or inductive"
            )

    args = parser.parse_args()
    
    # generate the filtration curves (saved to csv for easier handling)
    #create_curve()

    #
    list_of_df, y, column_names = load_curves(args)

    n_graphs = len(y)
    n_node_labels = list_of_df[0].shape[1]

    if args.method == "transductive":
        # standardize the size of the vector by forward filling then convert
        # to a vector representation. list_of_df now has length
        # n_node_labels
        list_of_df, index = index_train_data(list_of_df, column_names)
        X = []

        for graph_idx in range(n_graphs):
            graph_representation = []
            for node_label_idx in range(n_node_labels):
                graph_representation.extend(list_of_df[node_label_idx][graph_idx])
            X.append(graph_representation)
        
        run_rf(X, y)

    elif args.method == "inductive":
        X = list_of_df
        run_rf_inductive(X, y, column_names=column_names) 
