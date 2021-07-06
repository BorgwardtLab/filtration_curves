

We provide the sample code here. We do a bit of preprocessing to speed up the operations (defined in preprocessing). For the graphs, we assume that there are edge weights on the graph. If your data does not have edge weights, update the igraph version of it using some edge weight function, such that ```graph.es['weight']``` returns a list of edge weight values. This is critical for the code to run.

Please note that the MUTAG graphs are not the original dataset but rather the dataset with edge weights assigned using the Ricci curvature.

We presented two main graph descriptor functions: one using the node label histogram and one tracking the amount of connected components. 
