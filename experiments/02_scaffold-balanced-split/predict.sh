#!/bin/bash
#SBATCH --job-name=chemprop_baseline
#SBATCH --partition=shared
#SBATCH --time=05:00:00
#SBATCH --mem=16G
#SBATCH --output=experiments/02_scaffold-balanced-split/logs_%j.out #fix this!
#SBATCH --error=experiments/02_scaffold-balanced-split/logs_%j.err #fix this!

/gpfs/home/sugrani/bin/micromamba run -n pxr-chemprop chemprop predict \
    --test-path data/PXR_test.csv \
    --smiles-column SMILES \
    --model-path models/02_scaffold-balanced-split/model_0/best.pt \
    --preds-path results/02_scaffold-balanced_preds.csv