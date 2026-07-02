import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

for split, path in [("train", "data/PXR_train.csv"), ("test", "data/PXR_test.csv")]:
    df = pd.read_csv(path)
    mols = [Chem.MolFromSmiles(s) for s in df["SMILES"]]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
    fp_array = np.array(fps)
    pd.DataFrame(fp_array).to_csv(f"data/morgan_{split}.csv", index=False)
    print(f"{split}: {fp_array.shape}")
