

We provide the sample code here. We do a bit of preprocessing to speed up the operations (defined in preprocessing). For the graphs, we assume that there are edge weights on the graph. If your data does not have edge weights, update the igraph version of it using some edge weight function, such that ```graph.es['weight']``` returns a list of edge weight values. This is critical for the code to run.
