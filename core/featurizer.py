import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rd
import numpy as np
import h5py

from .file_io import DataReader
from .defaults import NBITS, RADIUS


class Featurizer(object):
    def __init__(self):
        self.nbits = NBITS
        self.radius = RADIUS
        self.ref_smiles = None

    def featurize(self, smiles, features_file=None):
        if features_file is None:
            return self.morgan_fingerprint(smiles)
        elif features_file[-3:] == "csv":
            return self.csv_lookup(features_file, smiles)
        elif features_file[-2:] == "h5":
            return self.molbert_lookup(features_file, smiles)
        else:
            print("Error: Featurizer could not resolve feature method")

    def molbert_lookup(self, features_file, smiles):
        with h5py.File(features_file, "r") as f:
            X, X_indxs = [], []
            if (
                self.ref_smiles is None
            ):  # Read smiles only once and store as class variable
                self.ref_smiles = {}
                read_smiles = [inp.decode("utf-8") for inp in f["Inputs"]]
                for i, s in enumerate(read_smiles):
                    self.ref_smiles[s] = i

            for s in smiles:
                if s in self.ref_smiles:
                    X_indxs.append(self.ref_smiles[s])
            X = [f["Values"][i] for i in X_indxs]

            return np.array(X, dtype=np.float32)

    def csv_lookup(self, features_file, smiles):
        X = []
        if self.ref_smiles is None:  # Read smiles only once and store as class variable
            self.ref_smiles = {}
            data_r = DataReader(features_file)
            read_smiles, features = data_r.read_all()
            for i, s in enumerate(read_smiles):
                self.ref_smiles[s] = features[i]

        for s in smiles:
            if s in self.ref_smiles:
                X.append(self.ref_smiles[s])
        return np.array(X, dtype=np.float32)

    def morgan_fingerprint(self, smiles):
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        fps = np.zeros((len(mols), self.nbits), dtype=np.float32)

        for i, m in enumerate(mols):
            if m is None:
                continue
            fp = rd.GetHashedMorganFingerprint(m, radius=self.radius, nBits=self.nbits)
            fps[i] = self.clip_sparse(fp)

        return np.array(fps, dtype=np.float32)

    def clip_sparse(self, vect):
        arr = np.zeros((self.nbits), dtype=np.float32)
        for i, v in vect.GetNonzeroElements().items():
            arr[i] = v if v < 255 else 255
        return arr

    def rdkit_version(self):
        return rdkit.__version__
