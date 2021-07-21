#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
import os
import glob
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder 
from pyper.representations import BettiCurve, make_betti_curve


def node_label_distribution(filtration, label_to_index):
    '''
    Calculates the node label distribution along a filtration. 

    Given a filtration from an individual graph, we calculate the node
    label histogram (i.e. the count of each unique label) at each step
    along that filtration, and returns a list of the weight of the filtration and
    its associated count vector. 

    Parameters
    ----------
    filtration : list
        A filtration of graphs
    label_to_index : mappable 
        A map between labels and indices, required to calculate the
        histogram.

    Returns
    -------
    D : list
        Label distributions along the filtration. Each entry is a tuple
        consisting of the weight of the filtration followed by a count
        vector.

    '''
    # Will contain the distributions as count vectors; this is
    # calculated for every step of the filtration.
    D = []

    for weight, graph in filtration:
        labels = graph.vs['label']
        counts = np.zeros(len(label_to_index))

        for label in labels:
            index = label_to_index[label]
            counts[index] += 1

        # The conversion ensures that we can serialise everything later
        # on into a `pd.series`.
        D.append((weight, counts.tolist()))

    return D


def filtration_curve_index(filtration_curves):
    '''
    Gets the union of all edge weights from the filtration curve. 

    Given the connected components filtration curves, gets the union of
    all edge weights. 

    Parameters
    ----------
    filtration_curves: list
        A list of the connected component filtrations.

    Returns
    -------
    full_index : index
        The full index of all curves.
    '''

    df_index = None
    for idx, curve in enumerate(tqdm(filtration_curves)):
        if idx == 0:
            df_index = curve._data.index
        else:
            df_index = df_index.union(curve._data.index)
    full_index = sorted(list(set(df_index)))

    return full_index


def reindex_filtration_curve(filtration_curve, new_index):
    '''
    Reindexes all connected components filtration curves with the union
    of all edge weights.

    Parameters
    ----------
    filtration_curve: pd.DataFrame 
        The connected component filtration curve.
    new_index: index
        The new index to use.

    Returns
    -------
    filtration_curve : pd.DataFrame
        The filtration curve that has been reindexed.

    ''' 
    # get rid of duplicates
    filtration_curve = BettiCurve(
            filtration_curve._data.loc[~filtration_curve._data.index.duplicated(keep="last")]
            )

    # reindex with full index
    filtration_curve = filtration_curve._data.reindex(
            new_index,
            method="ffill"
            ).fillna(0)
    return filtration_curve


def save_curves(
        source_path="../data/labeled_datasets/BZR_MD",
        output_path="../data/labeled_datasets/preprocessed_data/BZR_MD/"
        ):
    '''
    Creates the node label filtration curves. 

    Given a dataset of igraph graphs, we create a filtration curve using
    the node label histogram. For each edge weight in the graph, we
    generate a node label histogram for the subgraph induced by all
    edges with weight less than or equal to that given edge weight. We
    save each graph as a csv, so that we only do this filtration step
    once, and can reuse the results later.

    Parameters
    ----------
    source_path: str 
        The path to the dataset, which are igraphs stored in a pickle
        file.
    output_path: str
        The path to the output directory where the filtrations will be
        saved as a csv (one graph per csv)

    Returns
    -------

    '''
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
    '''
    Loads the precomputed node label filtration curves from csv files. 

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


def create_metric_dict(metrics=["accuracy"]):
    '''
    Creates a dictionary that will store the metrics calculated in cross
    validation.

    Parameters
    ----------
    metrics: list
        Metrics that will be used to assess performance of the
        classifier.

    Returns
    -------
    metric_dict : dict 
        Empty dictionary with the metrics of interest as keys.

    '''
    metric_dict = {}
    for metric in metrics:
        metric_dict[metric] = []

    return metric_dict


def compute_fold_metrics(y_test, y_pred, metric_dict):
    '''
    Calculates the metrics of interest on the classifier and updates the
    dictionary with the values.

    Given the true values (y_test) and the predicted values of
    a classifier (y_pred), this function computes the metrics of
    interest and updates the current dictionary (metrics_dict) with the value on
    the given fold.

    Parameters
    ----------
    y_test: array-like
        True values of the test data 
    y_pred: array-like 
        Predicted values from the classifier
    metric_dict: dict
        Dictionary containing the metric of interest and the values so
        far computed on previous folds
    
    Returns
    -------
    metric_dict : dict 
        Updated dictionary values containing the metric of interest and
        its value computed on the current fold

    '''
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, 1]
    
    # calculate accuracy 
    accuracy = accuracy_score(y_test, y_pred)
    
    # update dictionary values
    metric_dict["accuracy"].append(accuracy)

    return metric_dict


def update_iteration_metrics(fold_metrics, iteration_metrics):
    '''
    Updates the dictionary of iteration metrics with the average of the
    fold metrics.

    Updates the list of iteration-level metric results of the
    classifier by appending the mean of the fold metrics.  This is
    necessary when running multiple iterations of k-fold cross
    validation.

    Parameters
    ----------
    fold_metrics: dict  
        A dictionary containing the metrics and their results evaluated
        on the individual folds of cross validation.
    iteration_metrics: dict 
        A dictionary containing the metrics and the mean results from
        all folds of k-fold cross validation.

    Returns
    -------
    iteration_metrics: dict 
        An updated dictionary of the iteration-level metrics, including
        the current iteration.

    '''
    for metric in ["accuracy"]:
        iteration_metrics[metric].append(np.mean(fold_metrics[metric]))

    return iteration_metrics


def print_iteration_metrics(iteration_metrics, f=None):
    '''
    Prints the mean and standard deviation of the metrics over all
    iterations.    

    Parameters
    ----------
    iteration_metrics: dict
        Dictionary of metrics and the mean accuracy on each iteration.
    f: str
        File name to save the results, if desired.

    Returns
    -------

    '''
    for metric in iteration_metrics:#
        mean = np.mean(iteration_metrics[metric]) * 100
        sdev = np.std(iteration_metrics[metric]) * 100
        if f is None:
            print(f'{metric}: {mean:2.2f} +- {sdev:2.2f}') 
        else:
            print(f'{metric}: {mean:2.2f} +- {sdev:2.2f}', file=f) 



