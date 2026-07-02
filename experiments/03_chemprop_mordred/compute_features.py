import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors

calc = Calculator(descriptors, ignore_3D=True)

for split, path in [("train", "data/PXR_train.csv"), ("test", "data/PXR_test.csv")]:
    df = pd.read_csv(path)
    mols = [Chem.MolFromSmiles(s) for s in df["SMILES"]]
    features = calc.pandas(mols)
    # Keep only numeric, drop columns with any NaN
    features = features.select_dtypes(include="number").dropna(axis=1)
    features.to_csv(f"data/mordred_{split}.csv", index=False)
    print(f"{split}: {features.shape}")
