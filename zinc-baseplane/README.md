## About

This repository contains a modified version of the code used by the [PlanE: Representation Learning over Planar Graphs paper](https://arxiv.org/abs/2307.01180). We adapt their framework to evaluate the performance of BasePlanE with homomorphism counts for the ZINC12k dataset without edge features (as presented in Section 5.1 of our paper). 

Note that most modifications were made to the `datasets/zinc.py`, `plane/models.py`, and `experiments/main.py` files. For refenece, the original PlanE repo can be found [here](https://github.com/ZZYSonny/PlanE/tree/ce2561bfae46248c3260ac91b4a59be5d0d1c9a1).

## Requirements
We provide all the dependencies in a conda environment. You can create the environment by running
```bash
conda env create -f environment.yml
```

You may also use [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) to speed up environment installation.

## Datasets
### Preparing Homomorphism Counts
We use the same homomorphism counts across all models for the ZINC dataset. Please download and unzip the `zinc-data.zip` file, which you can access [here](https://github.com/ejin700/hombasis-gnn/blob/main/hombasis-bench/data/zinc-data.zip), and move the homomorphism count files into the `.dataset_src` directory.

### Dataset Preparation
Before running the experiments, you will first need to prepare the dataset. Simply run the following command to download and preprocess the datasets used for this model. 

```bash
python3 -m preprocess.prepare
```

Note that the current dataset preparation file will only generate the dataset using the anchored Spasm* homomorphism counts, but to prepare the dataset for other count types, simply change the `count_type` parameter in the `get_dataset` function on line 17. The only supported count type arguments are: `[subgraph, homcount, spasm, anchoredSpasm]`, which correspond to the configurations used in Table 1 (Section 5.1) of our paper.

## Run experiments

We use [wandb](https://wandb.ai/) to track experiments. You can use our pre-defined configuration files by creating a sweep from the [experiments/config/real_world/zinc](experiments/config/real_world/zinc/) folder. To create a wandb sweep to run BasePlanE and reproduce our anchored Spasm* results from Table 1, run:

```bash
wandb sweep experiments/config/real_world/zinc/12k-noe-plane-anchoredSpasm.yaml
```

After creating a sweep, the command to launch the sweep will be shown on the command line output. They are usually of the form:

```bash
wandb agent <username>/<project>/<sweep_id>
```

Once the experiment is launched, you can find the result in the wandb dashboard. The training/validation/test metric is logged as `train`, `valid`, `test`.
