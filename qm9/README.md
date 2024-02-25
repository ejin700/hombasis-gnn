## About

This repository contains a modified version of the code used by the [Shortest Path Message Passing Neural Network (SP-MPNN) paper](https://arxiv.org/abs/2206.01003). We adapt their framework to evaluate the performance of R-GCN with homomorphism counts for the QM9 dataset (as presented in Section 5.2 and Appendix D.4 of our paper). 

Note that most modifications were made to the `src/utils/dataset_loader.py`, `src/utils/model_loader.py`, and `src/models/gcn.py` files. For refenece, the original SP-MPNN repo can be found [here](https://github.com/radoslav11/SP-MPNN).

## Requirements

The requirements for the Python environment can be found in ``requirements.txt``. The main packages that have been used are PyTorch, PyTorch Geometric, OGB (Open Graph Benchmark), Neptune (neptune.ai), and Numpy.

Running on a GPU is supported via CUDA.

Tested combination: Python 3.11.5 + PyTorch 2.1.2 + PyTorch Geometric 2.4.0.

## Datasets

### QM9

The QM9 dataset is provided in ``data/QM9``. The homomorphism counts of all connected graphs with up to 5 vertices for all graphs in the train, valid, and test set are contained in the `data/QM9/v5_homcounts.zip` file. This zip file contains a folder with three files: `train_homcounts.json`, `valid_homcounts.json`, and `test_homcounts.json`.

Before running the model training for R-GCN, please unzip the `data/QM9/v5_homcounts.zip` file, and move the contents into the `data/QM9` directory.

## Running

The script we use to run the experiments is ``src/main.py``. Note that the script should be run from inside the ``src`` directory, or mark it as Source Root.

The main parameters of the script are:

- ``--dataset`` the dataset we use. We use `QM9`.
- ``--model`` for the model used. We use `gcn` for R-GCN.
- ``--mode`` for the current task type. We use ``gr`` for Graph Regression. 

Additionally, some of the more useful configurable parameters are:

- ``--emb_dim`` for the embedding dimensionality.
- ``--batch_size`` for the batch size we use during training.
- ``--lr`` for the learning rate.
- ``--dropout`` for the dropout probability.
- ``--epochs`` for the number of epochs.
- ``--num_layers`` for the number of layers.
- ``--res_freq`` for the layer interval for residual connections.

QM9 specific parameters include:
- ``--specific_task`` for the integer id for the QM9 objective / task we want to predict.
- ``--nb_reruns`` for the number of repeats per task.

A detailed list of all additional arguments can be seen using the following command:

``python main.py -h``

## Example running 
In order to reproduce the results in Section 5.2 of the paper, run the following:

```bash
cd src
python main.py -d QM9 -m GCN --mode gr --res_freq 2 --batch_size 128 --emb_dim 128 --num_layers 8
```

To run 5 reruns of the model on only mu (the first property), use the following command:

```bash
cd src
python main.py -d QM9 -m GCN --mode gr --res_freq 2 --batch_size 128 --emb_dim 128 --num_layers 8 --nb_reruns 5 --specific_task 0 
```

## Neptune.ai

You can use [neptune.ai](https://neptune.ai) to track the progress, by specifying your project and token in ``src/config.ini``.  Leave the fields as ``...`` if you want to just run locally. Alternatively, you can manually set your project and api token in lines 35 and 36 of `src/main.py`.

## Testing R-GCN+Hom with Fully-Adjacent layer
We also tested a version of R-GCN that uses a fully-adjacent layer at the end in accordance with [Alon et al. (2021)](https://arxiv.org/abs/2006.05205). In order to reproduce those results (which are presented in Appendix D.4), replace the `src/main.py` and `src/models/gcn.py` files with the corresponding files in the `FA_files` directory, and use the commands above to run the model.