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


def create_curves(args):
    '''
    Creates the node label histogram filtration curves. 

    Creates a node label histogram filtration curve, either by loading
    the previously generated filtration curves, or by calling
    save_curves(), which will generate the curves and same them as a csv
    file.

    Parameters
    ----------
    args: dict 
        Command line arguments, used to determine the dataset 

    Returns
    -------
    list_of_df: list
        A list of node label filtration curves, each stored as
        a pd.DataFrame
    y: list
        List of graph labels, necessary for classification.
    column_names: list
        List of column names (i.e. each unique node label)

    '''
    # check if filtration curves are already saved. If not, generate
    # them and save them.
    if not os.path.exists("../data/labeled_datasets/" + args.dataset
    + "/"):
        save_curves()
    
    # load saved curves (faster processing)
    list_of_df, y, column_names = load_curves(args)
    
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
    list_of_df, y, column_names = create_curves(args)

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
