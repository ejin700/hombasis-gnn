# HomGIN (Modified GSN)

This repository contains a modified version of GSN that uses homomorphism basis counts instead of subgraph isomorphism counts as node features. This implementation is effectively GIN + Hom(v5), where Hom(v5) refers to the homomorphsm counts of all connected graphs with up to 5 vertices, inclusive. 

Note that most modifications were made to the `utils_ids.py`, `utils_data_gen.py`, and `utils.py` files. For reference, the original BREC GSN repo can be found [here](https://github.com/GraphPKU/BREC/tree/Release/GSN).

## Requirements

Please refer to [GSN](https://github.com/gbouritsas/GSN) for additional requirements. Note that graph-tool is **NOT** required, since we use homomorphism counts instead of subgraph counts. 

## Usages

### Arrange data files

The following set of instructions is to prepare the datafiles for the CFI graphs. For the other datafiles, please adjust the filenames accordingly.

1. Create `datasets/BREC/brec_cfi` folder
2. Copy the `brec_cfi.g6` file that was generated from the [customize](https://github.com/icml2024357/hombasis-gnn/tree/main/brec/customize) directory into the `datasets/BREC/brec_cfi` directory created above
3. Copy the corresponding `brec_cfi_v5_counts.json` homomorphism count datafile from the [Dropbox link](https://www.dropbox.com/scl/fi/zpnaa89ip8qqlwro5yhc4/brec_counts.zip?rlkey=fcpixcdht0ks4gdiiwlhw7smk&dl=0) into the same `datasets/BREC/brec_cfi` directory

### Run evaluation

To reproduce best result, run:

```bash
python test_BREC_search.py
```

To reproduce other results for different data subsets, please change the corresponding code in `test_BREC_search.py` and `test_BREC.py`:
- Update the `DATASET_NAME` variable in `test_BREC_search.py`
- Update the `part_dict` variable in `test_BREC.py`
