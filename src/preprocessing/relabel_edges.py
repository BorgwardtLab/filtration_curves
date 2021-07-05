
#!/usr/bin/env python3
#

import os
import argparse
import glob
from tqdm import tqdm
from collections import Counter

import numpy as np 
import igraph as ig 
import networkx as nx
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler

from GraphRicciCurvature.OllivierRicci import OllivierRicci


def relabel_edges_with_curvature(graphs):
    """ relables edges with O.F. curvature. Note this requires
    converting to networkx and back. Currently this is simple since we
    have no node or edge labels. But would need to reconsider if we
    extend this to labeled graphs. """
    #
    # save graph label and then convert to networkx (nx loses label)
    y = [graph['label'] for graph in graphs]
    try:
        node_labels = [g.vs['label'] for g in graphs]
    except:
        print("no node labels") 
    graphs = [ig_to_nx(graph) for graph in tqdm(graphs)]
    
    # compute curvature and append as edge weight
    graphs = [compute_curvature(graph) for graph in tqdm(graphs)]
    
    # convert back to igraph
    graphs = [ig.Graph.from_networkx(graph) for graph in tqdm(graphs)]
    
    # add graph label and node labels back
    for idx, label in enumerate(y):
        graphs[idx]['label'] = label
        try:
            graphs[idx].vs['label'] = node_labels[idx]
        except:
            print("no node labels") 
    return(graphs)
    
    
def compute_curvature(graph):
    """ compute curavture and relabel edges in the graph """

    orc = OllivierRicci(graph, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    for u,v,e in graph.edges(data=True):
        graph[u][v]['weight']= orc.G[u][v]["ricciCurvature"]
    
    return(graph)


def ig_to_nx(graph):
    edge_list = graph.get_edgelist()
    G = nx.Graph(edge_list)
    return(G)


def relabel_nodes(graphs):
    for idx, graph in enumerate(graphs):
        graph.vs['label'] = [v.degree() for v in graph.vs]
    return(graphs)



