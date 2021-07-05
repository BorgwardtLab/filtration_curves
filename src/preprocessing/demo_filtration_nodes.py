#!/usr/bin/env python3
#
# Provides a simple demonstration of evaluating functions alongside of
# a filtration and collating the results. This demo uses node labels.

import argparse
import glob
import itertools
import os
import time
import csv

import igraph as ig
import numpy as np
import pandas as pd

from filtration import filtration_by_edge_attribute
from tqdm import tqdm

import relabel_edges as r
#import wl_relabeling as w


def label_distribution(filtration, label_to_index):
    '''
    Calculates the node label distribution of a filtration, using a map
    that stores index assignments for labels.

    :param filtration: A filtration of graphs
    :param label_to_index: A map between labels and indices, required to
    calculate the histogram.

    :return: Label distributions along the filtration. Each entry is
    a tuple consisting of the weight of the filtration followed by a
    count vector.
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('DIRECTORY', help='Input directory')
    parser.add_argument('PREFIX', help='Output prefix')
    parser.add_argument("-re", '--RELABEL_EDGES', 
            help='Set edge weights',
            action='store_true')

    args = parser.parse_args() 
    dataset = args.DIRECTORY.split("/")[-2]
    
    #RELABEL_EDGES = args.RELABEL_EDGES
    #RELABEL_NODES = False
    
    start = time.process_time()
    # Get all filenames; this ensures that the shell does *not* complain
    # about the length of the argument list.
    
    filenames = sorted(
        glob.glob(os.path.join(args.DIRECTORY, '*.pickle'))
        )
    
    graphs = [
        ig.read(filename, format='picklez') for filename in tqdm(filenames)
    ]

    for graph in graphs:
        graph.es['weight'] = [[e['attribute']] for e in graph.es]

    #if RELABEL_EDGES == True:
    #    graphs = r.relabel_edges(filenames=filenames, method="euclidean")
    #if RELABEL_NODES:
    #    graphs = r.relabel_nodes(graphs)

    # Get all potential node labels to make sure that the distribution
    # can be calculated correctly later on.
    node_labels = sorted(set(
        itertools.chain.from_iterable(graph.vs['label'] for graph in graphs)
    ))

    label_to_index = {
        label: index for index, label in enumerate(sorted(node_labels))
    }
    
    filtrated_graphs = [
        filtration_by_edge_attribute(
            graph,
            attribute='attribute',
            delete_nodes=True,
            stop_early=False
        )
        for graph in tqdm(graphs)
    ]
    
    # Create a data frame for every graph and store it; the output is
    # determined by the input filename, albeit with a new extension.
    for index, filtrated_graph in enumerate(tqdm(filtrated_graphs)):
        
        df = pd.DataFrame(columns=['graph_label, weight'].extend(node_labels))

        distributions = label_distribution(filtrated_graph, label_to_index)

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

        #output_name = os.path.basename(filenames[index])
        #output_name = os.path.splitext(output_name)[0] + '.csv'
        #output_name = os.path.join(args.PREFIX, output_name)

        #os.makedirs(args.PREFIX, exist_ok=True)

        #df.to_csv(output_name, index=False)

    end = time.process_time()
    print(dataset, end-start)

