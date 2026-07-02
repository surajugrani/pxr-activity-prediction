import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr, kendalltau
import pickle
import os
import json

# Load data
train_df = pd.read_csv("data/PXR_train.csv")
X = pd.read_csv("data/mordred_train_selected.csv")
y = train_df["pEC50"]

# Load ChemProp split
with open("models/01_chemprop_baseline/splits.json") as f:
    splits = json.load(f)

train_idx = splits[0]["train"]
val_idx = splits[0]["val"]

X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# Train RF
rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
preds = rf.predict(X_val)
mae = mean_absolute_error(y_val, preds)
r2 = r2_score(y_val, preds)
spear = spearmanr(y_val, preds).statistic
kendall = kendalltau(y_val, preds).statistic

print(f"MAE: {mae:.4f}")
print(f"R2: {r2:.4f}")
print(f"Spearman: {spear:.4f}")
print(f"Kendall: {kendall:.4f}")

# Save model
os.makedirs("models/03_chemprop_mordred", exist_ok=True)
with open("models/03_chemprop_mordred/rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)
    
# Save val predictions
val_preds_df = pd.DataFrame({
    "Molecule Name": train_df.iloc[val_idx]["Molecule Name"].values,
    "y_true": y_val.values,
    "y_pred": preds
})
val_preds_df.to_csv("models/03_chemprop_mordred/rf_val_predictions.csv", index=False)

# Save performance
with open("models/03_chemprop_mordred/rf_val_results.txt", "w") as f:
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"R2: {r2:.4f}\n")
    f.write(f"Spearman: {spear:.4f}\n")
    f.write(f"Kendall: {kendall:.4f}\n")