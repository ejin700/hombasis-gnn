## About

This folder contains all the scripts needed to prepare the separate BREC dataset files used in the experiments. 

## Usages

We have divided the BREC dataset into four different groups/files, which we then train and evaluate separately:
1. `dataset_v3_s4.py` contains the code to generate the Basic, Regular, and Extension graphs
2. `cfi.py` contains the code to generate the CFI graphs
3. `4vtx.py` contains the code to generate the 4-vertex condition graphs
4. `dr.py` contains the code to generate the Distance regular graphs

To produce the processed dataset file for basic, regular, and extension graphs, run:
```bash
python dataset_v3_s4.py
```
The resulting dataset file will appear in the `Data/processed` directory. To produce the processed dataset files for the other graph types, please change the corresponding dataset file name above. 

### File Types
Note that HomGIN (adapted from GSN) and PPGN use different data file extensions. HomGIN requires .g6 dataset files, whereas PPGN requires .npy files. In order to adjust for file type, please change the outputs to the `processed_file_names` and `process` functions in each file accordingly. 
