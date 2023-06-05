''' Script to run the filtration curves using connected components. ''' 


import argparse
import time
import math

import numpy as np
import pandas as pd
from scipy import stats
import igraph as ig

from sklearn.preprocessing import LabelEncoder

from rf import *
from utils import *
from preprocessing import relabel_edges as r

from pyper.representations import BettiCurve, make_betti_curve
import pyper.persistent_homology as ph



def create_curves(args):
    '''
    Creates the connected component filtration curves. 

    Uses the persistence diagram to create a filtration curve using the
    count of connected components. We use the BettiCurve function from
    the pyper package to do this.

    Parameters
    ----------
    args: dict 
        Command line arguments, used to determine the dataset 

    Returns
    -------
    filtration_curves: list
        A list of connected component filtration curves.
    y: list
        List of graph labels, necessary for classification.

    '''
    dataset = args.dataset
    file_path = "../data/unlabeled_datasets/" + dataset + "/"
    
    # This section is the normal section to load the graphs.
    filenames = sorted(
        glob.glob(os.path.join(file_path, '*.pickle'))
        )
    
    graphs = [
        ig.read(filename, format='picklez') for filename in tqdm(filenames)
        ]
    
    y = [graph['label'] for graph in graphs]
    
    # add the vertex creation weight as the node degree
    for graph in tqdm(graphs):
        graph.vs["weight"] = [graph.degree(v) for v in graph.vs]
    diagrams = [
            ph.calculate_persistence_diagrams(graph, 
            vertex_attribute="weight",
            edge_attribute="weight") for graph in tqdm(graphs)
        ]
    
    creation_destruction_pairs = [a._pairs for a,b in diagrams]
    
    # use the function from pyper to create the curve for us. We use the
    # betti curve for the calculation, but our instance is a more
    # general form since we do not make the same assumptions..
    filtration_curves = [make_betti_curve(a) for a,b in tqdm(diagrams)]

    return filtration_curves, y


if __name__ == "__main__":
    start = time.process_time()
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
            '--dataset', 
            help='Input file(s)', 
            )

    parser.add_argument(
            '--method',
            default="transductive",
            type=str,
            help="transductive or inductive"
            )
    
    args = parser.parse_args()
    
    # get filtration curves
    filtration_curves, y = create_curves(args)

    # relabel y
    y = LabelEncoder().fit_transform(y)

    if args.method == "transductive":
        # get the union of all indexes of the filtration curve
        new_index = filtration_curve_index(filtration_curves)

        # reindex each betti curve with the full index to standardize the size
        filtration_curves = [reindex_filtration_curve(b, new_index) for b in
            tqdm(filtration_curves)]

        # convert to numpy 
        filtration_curves = [i.to_numpy() for i in tqdm(filtration_curves)]
    
        # run the random forest
        run_rf(filtration_curves, y, n_iterations=10)

    elif args.method == "inductive":
        # format the curves as a dataframe
        filtration_curves = [pd.DataFrame(i._data) for i in tqdm(filtration_curves)]

        # get the column names (just a single one here)
        column_names = filtration_curves[0].columns.tolist()

        # run the random forest
        run_rf_inductive(
                filtration_curves, 
                y,
                column_names=column_names
                )
