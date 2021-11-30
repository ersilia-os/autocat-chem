from .base import AutoCatBase
from catboost import CatBoostRegressor
import onnxruntime
import numpy as np


class AutoCatPredictor(AutoCatBase):
    def __init__(self, features_file=None):
        AutoCatBase.__init__(self)
        self.model = ""
        self.features_file = features_file

    def predict(self, smiles, scaler):
        X = self.featurizer.featurize(smiles, features_file=self.features_file)
        if self.model == "onnx":
            preds = self.session.run(None, {"features": X.astype(np.float32)})
            preds = np.array(preds, dtype=np.float32)
            preds_arr = np.reshape(preds, (X.shape[0], preds.shape[-1]))
        elif type(self.model) == CatBoostRegressor:
            preds = self.model.predict(X)
            if len(preds.shape) == 1:
                preds_arr = np.reshape(preds, (X.shape[0], 1))
            else:
                preds_arr = np.reshape(preds, (X.shape[0], preds.shape[-1]))

        rescaled_preds = scaler.reverse_scale(preds_arr)
        return np.array(rescaled_preds, dtype=np.float32)

    def load_onnx(self, file_path):
        self.session = onnxruntime.InferenceSession(file_path)
        self.model = "onnx"

    def load_cbm(self, file_path):
        self.model = CatBoostRegressor()
        self.model.load_model(file_path)

    def set_model(self, model):
        self.model = model
