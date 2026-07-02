#!/bin/bash
#SBATCH --job-name=chemprop_baseline
#SBATCH --partition=shared
#SBATCH --time=05:00:00
#SBATCH --mem=16G
#SBATCH --output=experiments/02_scaffold-balanced-split/logs_%j.out #fix this!
#SBATCH --error=experiments/02_scaffold-balanced-split/logs_%j.err #fix this!

/gpfs/home/sugrani/bin/micromamba run -n pxr-chemprop chemprop train \
    --data-path data/PXR_train.csv \
    --smiles-column SMILES \
    --target-columns pEC50 \
    --split-type scaffold_balanced \
    --save-dir models/02_scaffold-balanced-split \
    --accelerator cpu