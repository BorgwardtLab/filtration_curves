#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
import os

from sklearn.metrics import accuracy_score

def create_metric_dict(
        metrics=["accuracy"]
        ):
    """ creates and returns a dict with an empty list for each of the included
    metrics """

    metric_dict = {}
    for metric in metrics:
        metric_dict[metric] = []

    return(metric_dict)


def compute_fold_metrics(y_test, y_pred, metrics_dict):
    """ computes the stardard metrics and updates the dictionary
    containing all metric results. Input is true y, predicted y, and
    current metrics dict """
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, 1]
    
    # calculate accuracy 
    accuracy = accuracy_score(y_test, y_pred)
    
    # update dictionary values
    metrics_dict["accuracy"].append(accuracy)

    return(metrics_dict)


def update_fold_metrics(all_fold_metrics, single_fold_metrics):
    ''' At the end of hyperparam optimization, adds the single fold
    metrics to the all fold metrics dictionary'''

    for k in single_fold_metrics:
        all_fold_metrics[k].append(single_fold_metrics[k])
    return(all_fold_metrics)


def update_iteration_metrics(fold_metrics, iteration_metrics):
    """ Add average of fold metrics to iteration metrics """

    for metric in ["accuracy"]:
        iteration_metrics[metric].append(np.mean(fold_metrics[metric]))

    return(iteration_metrics)


def print_iteration_metrics(iteration_metrics, f=None):
    """ print the mean and sd of each of the metrics averaged over all
    iterations """

    for metric in iteration_metrics:#
        mean = np.mean(iteration_metrics[metric]) * 100
        sdev = np.std(iteration_metrics[metric]) * 100
        if f is None:
            print(f'{metric}: {mean:2.2f} +- {sdev:2.2f}') 
        else:
            print(f'{metric}: {mean:2.2f} +- {sdev:2.2f}', file=f) 



