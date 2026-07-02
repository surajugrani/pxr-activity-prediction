import pandas as pd
import pickle

test_df = pd.read_csv("data/PXR_test.csv")
X_test = pd.read_csv("data/morgan_test.csv")

with open("models/04_rf_morgan/rf_model.pkl", "rb") as f:
    rf = pickle.load(f)

preds = rf.predict(X_test)

results = pd.DataFrame({
    "Molecule Name": test_df["Molecule Name"],
    "SMILES": test_df["SMILES"],
    "pEC50": preds
})
results.to_csv("results/04_rf_morgan_preds.csv", index=False)
print("Done")
