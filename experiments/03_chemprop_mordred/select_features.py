import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

# Load precomputed descriptors
train_feats = pd.read_csv("data/mordred_train.csv")
test_feats = pd.read_csv("data/mordred_test.csv")
y = pd.read_csv("data/PXR_train.csv")["pEC50"]

# Step 1: Keep only columns with no NaN in train
valid_cols = train_feats.columns[train_feats.notna().all()]
train_feats = train_feats[valid_cols]

# Step 2: Remove low variance
selector = VarianceThreshold(threshold=0.01)
selector.fit(train_feats)
train_feats = pd.DataFrame(selector.transform(train_feats),
                           columns=train_feats.columns[selector.get_support()])

# Step 3: Remove highly correlated, keep one with higher correlation to pEC50
corr_with_y = train_feats.corrwith(y).abs()
corr_matrix = train_feats.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = set()
for col in upper.columns:
    correlated = upper.index[upper[col] > 0.9].tolist()
    for other in correlated:
        if col not in to_drop and other not in to_drop:
            if corr_with_y[col] < corr_with_y[other]:
                to_drop.add(col)
            else:
                to_drop.add(other)

train_feats = train_feats.drop(columns=to_drop)
final_cols = train_feats.columns

# Apply same columns to test
test_feats = test_feats[final_cols]

# Save
train_feats.to_csv("data/mordred_train_selected.csv", index=False)
test_feats.to_csv("data/mordred_test_selected.csv", index=False)
print(f"train: {train_feats.shape}, test: {test_feats.shape}")
