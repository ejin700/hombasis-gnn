# Towards Better Evaluation of GNN Expressiveness with BREC Dataset

## About

This repository contains a modified subset of the codebase provided by the following paper: [Towards Better Evaluation of GNN Expressiveness with BREC Dataset](https://arxiv.org/abs/2304.07702). For requirements and original implementation details, please refer to the original [BREC repository](https://github.com/GraphPKU/BREC).

## Usages

### Requirements

Tested combination: Python 3.11.5 + PyTorch 2.1.2 + PyTorch Geometric 2.4.0.

Other required Python libraries included: numpy, networkx, loguru.

For reproducing other results, please refer to the corresponding requirements for additional libraries.

### <span id="preparation">Data Preparation</span>

#### Step 1: Prepare BREC dataset files

The 400 pairs of graphs are from four categories: Basic, Regular, Extension, CFI. 4-vertex condition and distance regular graphs are further split from Regular graphs as separate categories. The "category-id_range" dictionary is as follows:

```python
  "Basic": (0, 60),
  "Regular": (60, 160),
  "Extension": (160, 260),
  "CFI": (260, 360),
  "4-Vertex_Condition": (360, 380),
  "Distance_Regular": (380, 400),
```

We divide the entire BREC dataset into four different groups/files, which we then train and evaluate separately:
1. Basic, Regular, and Extension graphs
2. CFI graphs
3. 4-vertex condition graphs
4. Distance regular graphs

Instructions on how to generate these four BREC dataset files can be found in the `customize` folder.

#### Step 2: Download corresponding homomorphism count data

The homomorphism counts for all 5-vertex connected graphs for each BREC dataset file can be downloaded at the following [link](https://www.dropbox.com/scl/fi/zpnaa89ip8qqlwro5yhc4/brec_counts.zip?rlkey=fcpixcdht0ks4gdiiwlhw7smk&dl=0).

Instructions on how and where to arrange these homomorphism count files for each model are in the respective directories.