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
from preprocessing import relabel_edges as r

from pyper.representations import BettiCurve, make_betti_curve
import pyper.persistent_homology as ph



def create_curve(args):
    """ Use the persistence diagram on all graphs and return
    a list of (creation, destruction) tuples per graph. """
    dataset = args.dataset
    file_path = "../preprocessed_data/unlabeled_datasets/" + dataset + "/"
    
    # This section is the normal section to load the graphs.
    filenames = sorted(
        glob.glob(os.path.join(file_path, '*.pickle'))
        )
    
    graphs = [
        ig.read(filename, format='picklez') for filename in tqdm(filenames)
        ]
    
    y = [graph['label'] for graph in graphs]
    
    # add the vertex creation weight as the node degee
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
    creation_destruction_pairs = [make_betti_curve(a) for a,b in tqdm(diagrams)]

    return(creation_destruction_pairs, y)



def filtration_curve_index(filtration_curves):
    """ Gets the union of all Betti curve indexes. """
    df_index = None
    for idx, curve in enumerate(tqdm(filtration_curves)):
        if idx == 0:
            df_index = curve._data.index
        else:
            df_index = df_index.union(curve._data.index)

    return(sorted(list(set(df_index))))



def reindex_filtration_curve(filtration_curve, new_index):
    """ Reindex a filtration curve to have full index of the dataset. This
    requires getting rid of duplicates, since there seem to be some in
    the IMDB-BINARY dataset. """

    # get rid of duplicates
    filtration_curve = BettiCurve(
            filtration_curve._data.loc[~filtration_curve._data.index.duplicated(keep="last")]
            )

    # reindex with full index
    filtration_curve = filtration_curve._data.reindex(
            new_index,
            method="ffill"
            ).fillna(0)
    return(filtration_curve)
    


if __name__ == "__main__":
    start = time.process_time()
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
            '--dataset', 
            help='Input file(s)', 
            )
    
    args = parser.parse_args()
    
    # get filtration curves
    filtration_curves, y = create_curve(args)
    
    # relabel y
    y = LabelEncoder().fit_transform(y)


    # get the union of all indexes of the betti curve
    new_index = filtration_curve_index(filtration_curves)

    # reindex each betti curve with the full index to standardize the size
    filtration_curves = [reindex_filtration_curve(b, new_index) for b in
            tqdm(filtration_curves)]

    # convert to numpy 
    filtration_curves = [i.to_numpy() for i in tqdm(filtration_curves)]
    
    # run the random forest
    run_rf(filtration_curves, y, args, n_iterations=10)
