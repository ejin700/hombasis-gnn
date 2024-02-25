# PPGN Reproduction

## Requirements

Please refer to [PPGN](https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch)

## Usages

### Arrange data files

The following set of instructions is to prepare the datafiles for the CFI graphs. For the other datafiles, please adjust the filenames accordingly.

1. Create `Data/raw` folder
2. Copy the `brec_cfi.g6` file that was generated from the [customize](https://github.com/icml2024357/hombasis-gnn/tree/main/brec/customize) directory into the `Data/raw` directory created above
3. Copy the corresponding `brec_cfi_v5_counts.json` homomorphism count datafile from the [Dropbox link](https://www.dropbox.com/scl/fi/zpnaa89ip8qqlwro5yhc4/brec_counts.zip?rlkey=fcpixcdht0ks4gdiiwlhw7smk&dl=0) into the same `Data/raw` directory

Note that the program will process the raw data and store it in a newly created `Data/no_param/processed` folder. For future runs, the model will try to load the pre-processed data that is saved here. Therefore, if you change the data file or make any other adjustments to the input data, remember to delete the processed data files stored in `Data/no_param/processed` in order to force the generation of the new dataset. If you do not do this, the model will use the pre-saved data from the previous run, and the results may either be incorrect or the model will error out.

### Run evaluation

To reproduce best result on PPGN, run:

```bash
cd main_scripts
python test_BREC_search.py
```

The configs are stored in configs/BREC.json

To reproduce other results for different data subsets, please change the corresponding code in `BRECDataset_v3.py` and `test_BREC.py`:
- Update the `DATA_SPLIT_NAME` variable in `BRECDataset_v3.py`
- Update the `part_dict` variable in `main_scripts/test_BREC.py`
