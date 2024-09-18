# Homomorphisms Counts for Graph Neural Networks: All About That Basis

This is the official code base for the ICML 2024 [paper](https://arxiv.org/abs/2402.08595): *Homomorphism Counts for Graph Neural Networks: All About That Basis*.

This repository is divided into four sub-projects:
1. The subdirectory `brec` is a partial clone of [BREC](https://github.com/GraphPKU/BREC/tree/966b5ed5c27bf6372ea081f90ef8c8b2a3342ebc) by [Wang & Zhang (2023)](https://arxiv.org/abs/2304.07702). This project can be used to reproduce the BREC experiements from Section 5.4 of the paper. 
2. The subdirectory `hombasis-bench` is a PyTorch re-implementation of some of the models that were benchmarked by [Dwivedi et al. (2023)](https://arxiv.org/abs/2003.00982) and [Hu et al. (2020)](https://arxiv.org/abs/2005.00687). This project can be used to reproduce the ZINC and COLLAB experiments from Sections 5.1 and 5.3 of the paper, respectively. 
3. The subdirectory `pact` provides the code for all homomorphism pattern counting that was used in this paper. The official repo for PACT (Pattern Counting Toolkit) has also been made public [here](https://github.com/dbai-tuw/pact).
4. The subdirectory `qm9` is a modified clone of [SP-MPNN](https://github.com/radoslav11/SP-MPNN/tree/main) by [Abboud et al. (2022)](https://arxiv.org/abs/2206.01003). This project can be used to reproduce the QM9 experiments from Section 5.2 of the paper.
5. The subdirectory `zinc-baseplane` is a modified clone of [PlanE](https://github.com/ZZYSonny/PlanE/tree/ce2561bfae46248c3260ac91b4a59be5d0d1c9a1) by [Dimitrov et al. (2023)](https://arxiv.org/abs/2307.01180). This project can be used to reproduce the BasePlanE experiments on ZINC from Section 5.1 of the paper.

For installation and dependency information for each sub-project, please refer to the README files in the respective directories.

## Cite
If you make use of this code, or its accompanying [paper](https://arxiv.org/abs/2402.08595), please cite this work as follows:
```
@inproceedings{JBCL-ICML24,
  title = "Homomorphism Counts for Graph Neural Networks: All About That Basis",
  author = "Emily Jin and Michael Bronstein and {\.I}smail {\.I}lkan Ceylan and Matthias Lanzinger",
  year = "2024",
  booktitle = "Proceedings of Fourty-first International Conference on Machine Learning (ICML)",
  url = "https://arxiv.org/abs/2402.08595",
}
```
