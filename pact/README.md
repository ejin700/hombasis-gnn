# PACT --  A Pattern Counting Toolkit
This is a toolkit for exact pattern counting, in particular, PACT currently supports exact subgraph counting for undirected and directed graphs.

Once everything has been set up (see below) we recommend to run
```sh
pipenv run jupyter notebook
```
from the main project directory and open the `Tutorial 1` notebook for an introduction on how to use Mobius.

## Installation
Necessary dependencies that are not installed automatically:
    - A working C compile chain
    - [go](https://go.dev/)
    - python 3.10 and pipenv
 
The rudimentary `setup.sh` script should set everything else up.

### Pynauty issues
The standard installation of `pynauty` doens't seem to work everywhere at the moment. If you are having problems with functionality that depends on nauty you can try the following in your work dir to reinstall pynauty in a more reliable way.
```sh
pipenv shell
pip uninstall pynauty
pip install --no-binary pynauty pynauty
```

## Usage

Code for homomorphism counting for ZINC, QM9, and BREC is located in the `homcount_preprocessing.ipynb` notebook. A more efficient implementation of counting for the COLLAB dataset is located in the `collab_count.ipynb` notebook, and `collab_K5_count.ipynb` contains the code for counting specifically 5-cliques for COLLAB.

### Data and files

The `homcount_preprocessing.ipynb` notebook requries 2 files:
1. A homomorphism basis file for the pattern(s) of interest
2. A data file containing the host graph(s) to perform counting in

All basis files used in the paper (as well as some additional ones) are located in the `bases` folder. 

The graph data files are formatted as json files with the following structure:

```
{
    <graph_idx>: {
        'edge_index': <graph_edge_index>,
    }
    ...
}
```

We provide the input file for the ZINC dataset in `zinc/zinc12k.zip` for reference. To use this file to perform homomorphism counts on ZINC, please unzip this file first.

The output files that containing homomorphism counts are also json files with the following structure:

```
{
    <graph_idx>: {
        'edge_index': <graph_edge_index>,
        'homcounts': {
            <vertex_idx>: <homomorphism_basis_count_list>,
            ...
        }
        'coefficients': <alpha_coefficients>
    }
    ...
}
```

Note that the "coefficients" field is only for when the basis given is for a sigle pattern (ie the 8 cycle basis). It is not provided for n-vertex homomorphism counting (that is used in QM9 and BREC).

We provide all homomorphism count files for ZINC, COLLAB, QM9, and BREC in their respective directories. 