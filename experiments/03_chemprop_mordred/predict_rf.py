import pandas as pd
import pickle

X_test = pd.read_csv("data/mordred_test_selected.csv")

with open("models/03_chemprop_mordred/rf_model.pkl", "rb") as f:
    rf = pickle.load(f)

preds = rf.predict(X_test)

test_df = pd.read_csv("data/PXR_test.csv")
results = pd.DataFrame({
    "Molecule Name": test_df["Molecule Name"],
    "SMILES": test_df["SMILES"],
    "pEC50": preds
})
results.to_csv("results/03_rf_preds.csv", index=False)
print("Done")