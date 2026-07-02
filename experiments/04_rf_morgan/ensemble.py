import pandas as pd

chemprop_preds = pd.read_csv("results/01_chemprop_baseline_preds.csv")
rf_preds = pd.read_csv("results/04_rf_morgan_preds.csv")

ensemble_preds = pd.DataFrame({
    "Molecule Name": chemprop_preds["Molecule Name"],
    "SMILES": chemprop_preds["SMILES"],
    "pEC50": (chemprop_preds["pEC50"] + rf_preds["pEC50"]) / 2
})

ensemble_preds.to_csv("results/04_ensemble_morgan_preds.csv", index=False)
print("Done")
