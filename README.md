# Filtration Curves for Graph Representation

This repository provides the code from the KDD'21 paper *Filtration
Curves for Graph Representation.* 

## Dependencies

We used [poetry](https://python-poetry.org/) to manage our dependencies.
Once poetry is installed on your computer, navigate to the directory
containing this code and type `poetry install` which will install all of
the necessary dependencies (provided in the `pyproject.toml` file.

## Data

We've provided sample data to work with to show how the method works out
of the box, provided in the `data` folder. Our method works with graphs
using `igraph`, and requires that the graphs have an edge weight (e.g.,
all weights in an `igraph` graph would be listed using the command `graph.es['weight']`. The BZR\_MD dataset had edge weights already, and therefore we provided the original dataset; the MUTAG dataset did not have edge weights, so the data provided has edge weights added (using the Ricci curvature).

If your graphs do not have an edge weight, there are numerous ways to
calculate them, which we detail in the paper. An example of how we added edge weights can be found in the `preprocessing/label_edges.py` file. 

## Method and Expected Output

In our work, we used two main graph descriptor functions: one using the node label histogram and one tracking the amount of connected components. There is a file for each; but please note that the node label histogram requires that the graph has node labels.

To run the node label histogram filtration curve, navigate to the `src`
folder and type the following command into the terminal:

```bash
$ poetry run python node_label_histogram_filtration_curve.py --dataset BZR_MD
```

This should return the following result in the command line: `accuracy: 75.61 +- 1.13`.
 
To run the connected components filtration curve (using the Ricci
curvature), navigate to the `src`
folder and type the following command into the terminal:

```bash
$ poetry run python connected_components_filtration_curve.py --dataset MUTAG
```

This should return the following result in the command line: `accuracy: 87.31 +- 0.66`.
 
## Citing our work

Please use the following BibTeX citation when referencing our work:

```bibtex
@inproceedings{OBray21a,
    title        = {Filtration Curves for Graph Representation},
    author       = {O'Bray, Leslie and Rieck, Bastian and Borgwardt, Karsten},
    doi          = {10.1145/3447548.3467442},
    year         = 2021,
    booktitle    = {Proceedings of the 27th ACM SIGKDD International
                 Conference on Knowledge Discovery \& Data Mining~(KDD)},
    publisher    = {Association for Computing Machinery},
    address      = {New York, NY, USA},
    pubstate     = {inpress},
}
```
