import os

DATASET_NAME = "brec_cfi"
SEED = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for seed in SEED:
    script_base = (
        f" python test_BREC.py --mode BREC_test --dataset BREC --dataset_name {DATASET_NAME} --root_folder ./datasets "
        f"--id_type all_simple_graphs --induced True --model_name GSN_sparse --msg_kind general "
        f"--num_layers 4 --d_out 64 --wandb False --seed {seed} "
    )

    script = script_base + " --k 4 --id_scope global"

    print(script)
    os.system(script)
