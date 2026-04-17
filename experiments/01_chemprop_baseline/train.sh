#!/bin/bash
#SBATCH --job-name=chemprop_baseline
#SBATCH --partition=shared
#SBATCH --time=05:00:00
#SBATCH --mem=16G
#SBATCH --output=experiments/01_chemprop_baseline/logs_%j.out
#SBATCH --error=experiments/01_chemprop_baseline/logs_%j.err

/gpfs/home/sugrani/bin/micromamba run -n pxr-chemprop chemprop train \
    --data-path data/PXR_train.csv \
    --smiles-column SMILES \
    --target-columns pEC50 \
    --save-dir models/01_chemprop_baseline \
    --accelerator cpu