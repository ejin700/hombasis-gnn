## About

This repository contains a PyTorch re-implementation of some of the models that were benchmarked by [Dwivedi et al. (2023)](https://arxiv.org/abs/2003.00982) for the ZINC dataset. It also contains an extention of the code provided by [Wang & Zhang (2023)](https://arxiv.org/abs/2304.07702) for the ogb-COLLAB dataset. This codebase can be used to reproduce the results in Sections 5.1 and 5.3 of the paper.

For reference, the original code base for the Dwivedi et al. (2023) paper can be accessed [here](https://github.com/graphdeeplearning/benchmarking-gnns/tree/master), and the original code base for the Wang & Zhang (2023) paper can be accessed [here](https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/collab).

## Requirements
The main packages that have been used are PyTorch, PyTorch Geometric, OGB (Open Graph Benchmark), wandb (wandb.ai), and numpy.

Running on a GPU is supported via CUDA.

Tested combination: Python 3.11.5 + PyTorch 2.1.0 + PyTorch Geometric 2.4.0 + OGB 1.3.6.

## Datasets

### ZINC

All homomorphism count data for the ZINC dataset is contained in the `data/zinc-data.zip` file. Please unzip this file before running any of the experiments. 

### COLLAB

All homomorphism count data for the COLLAB dataset is contained in the `data/collab-data.zip` file. Please unzip this file before running any of the experiments. 

## Running

We use configuration files to run all the experiments for both ZINC and COLLAB, which can be found in the `config` directory. All experimental results are logged in the wandb dashboard.

In order to reproduce the ZINC results for a given config file, run:

```
python3 run-zinc.py -c <config_file> -group <wandb_group_name> -project <wandb_project_name> -seed <seed>
```

Note that you must specify names for `<wandb_group_name>` and `<wandb_project_name>`, which will then show up in the wandb tracking dashboard.


In order to reproduce the COLLAB results for a given config file, run:

```
python3 run-collab.py -c <config_file> -seed <seed>
```

We rerun each experiment 4 times, using seed values of 41, 95, 12, and 35. The results reported in the paper are the average of these four runs. 

