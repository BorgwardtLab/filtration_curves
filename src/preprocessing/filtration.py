#!/usr/bin/env python3

'''
Provides a module for calculating a filtration of a weighted graph. This
operation is based on the `igraph` graph library.
'''


import igraph as ig
import numpy as np


def filtration_by_edge_attribute(
        graph,
        attribute='weight',
        delete_nodes=False,
        stop_early=False):
    '''
    Calculates a filtration of a graph based on an edge attribute of the
    graph.

    :param graph: Graph
    :param attribute: Edge attribute name
    :param delete_nodes: If set, removes nodes from the filtration if
    none of their incident edges is part of the subgraph. By default,
    all nodes are kept.
    :param stop_early: If set, stops the filtration as soon as the
    number of nodes has been reached.

    :return: Filtration as a list of tuples, where each tuple consists
    of the weight threshold and the graph.
    '''

    weights = graph.es[attribute]
    weights = np.array(weights)
    
    if len(weights.shape) == 2 and weights.shape[1] == 1:
        weights = weights.squeeze()

    else:
        raise RuntimeError('Unexpected edge attribute shape')

    # Represents the filtration of graphs according to the
    # client-specified attribute.
    F = []

    n_nodes = graph.vcount()
    
    # print(len(weights))    
    if weights.size != 1: # hack to deal with funny graph that has a single edge and was gettnig 0-D errors
        weights = weights
        x = False
    else:
        weights = np.array([[weights]])
        x = True
        # weights = [weights]
    
    for weight in sorted(weights):
        
        # print(type(weight))
        if x:               # again part of the hack
            weight = weight[0]
        edges = graph.es.select(lambda edge: edge[attribute] <= weight)
        subgraph = edges.subgraph(delete_vertices=delete_nodes)

        # Store weight and the subgraph induced by the selected edges as
        # one part of the filtration. The client can decide whether each
        # node that is not adjacent to any edge should be removed or not
        # in the filtration (see above).
        F.append((weight, subgraph))

        # If the graph has been filled to the brim with nodes already,
        # there is no need to continue.
        if stop_early and subgraph.vcount() == n_nodes:
            break


    return F
