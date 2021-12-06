from .fitter import AutoCatFitter
from .predictor import AutoCatPredictor
from .file_io import DataReader
from .scaler import Scaler
from .defaults import BATCH_SIZE
from .base import AutoCatTrain
import numpy as np


class AutoCat(object):
    def __init__(self, reference_lib=None):
        self.scaler = Scaler()
        self.metrics = {}
        self.data_r = ""
        self.data_len = 0
        self.batch_size = BATCH_SIZE
        self.reference_lib = reference_lib

    def fit(
        self, data, optimise_time=3600, weight=False
    ):  # if file, expected to have header row
        smiles, targets = self.check_input(data)
        self.scaler.get_params(targets)
        training_params = AutoCatTrain().train_params(targets)

        if self.data_r == "" or self.data_len <= self.batch_size:
            self.fitter = AutoCatFitter(
                self.scaler,
                training_params=training_params,
                features_file=self.reference_lib,
            )
        else:
            self.fitter = AutoCatFitter(
                self.scaler,
                training_params=training_params,
                features_file=self.reference_lib,
                batch=True,
                data_r=self.data_r,
                batch_size=self.batch_size,
                data_len=self.data_len,
            )

        if weight:
            self.fitter.weight_labels(targets)

        if optimise_time > 0:
            self.fitter.optimise_search(smiles, targets, time_budget=optimise_time)

        self.metrics = self.fitter.fit(smiles, targets)
        return self.metrics

    def predict(self, data, smiles_col=0):
        if type(data) == np.ndarray:
            smiles = data
        elif type(data) == str:
            data_r = DataReader(data)
            smiles = data_r.read_smiles(smiles_col=smiles_col)

        if self.metrics != {}:  # If there is a fitter trained in this AutoCat object
            self.predictor = AutoCatPredictor(features_file=self.reference_lib)
            self.predictor.set_model(self.fitter.get_model())

        return self.predictor.predict(smiles, self.scaler)

    def save(self, file_path, as_onnx=False):
        file_name = file_path.split(".")
        if as_onnx:
            if self.y.shape[1] > 1:
                raise Exception("Multiregression models cannot be saved in onnx format")
            self.fitter.save_model(file_name[0] + ".onnx", "onnx")
        else:
            self.fitter.save_model(file_name[0] + ".cbm", "cbm")
        self.fitter.save_metrics(file_name[0] + "_metrics.json")
        self.fitter.save_weights(file_name[0] + "_weights.json")
        self.scaler.save(file_name[0] + "_scaler.json")

    def load(self, file_path):
        self.predictor = AutoCatPredictor(features_file=self.reference_lib)
        file_name = file_path.split(".")
        if file_name[-1] == "onnx":
            self.predictor.load_onnx(file_path)
        elif file_name[-1] == "cbm":
            self.predictor.load_cbm(file_path)
        self.scaler.load(file_name[0] + "_scaler.json")

    # TO DO save and load training params
    def retrain(self, model_path, data):
        file_name = model_path.split(".")
        self.scaler.load(file_name[0] + "_scaler.json")
        smiles, targets = self.check_input(data)
        training_params = AutoCatTrain().train_params(targets)

        if self.data_r == "" or self.data_len <= self.batch_size:
            self.fitter = AutoCatFitter(
                self.scaler,
                training_params=training_params,
                features_file=self.reference_lib,
            )
        else:
            self.fitter = AutoCatFitter(
                self.scaler,
                training_params=training_params,
                features_file=self.reference_lib,
                batch=True,
                data_r=self.data_r,
                batch_size=self.batch_size,
                data_len=self.data_len,
            )
        self.fitter.load_weights(file_name[0] + "_weights.json")

        self.metrics = self.fitter.fit(smiles, targets, retrain=model_path)
        return self.metrics

    def check_input(self, data):
        if type(data) == list:
            self.data_len = len(data[0])
            smiles = data[0]
            targets = data[1]

        elif type(data) == str:
            self.data_r = DataReader(data)
            self.data_len = self.data_r.read_length()
            if self.data_len <= self.batch_size:
                smiles, targets = self.data_r.get_fold(0, self.data_len)
            else:
                smiles, targets = self.data_r.get_fold(0, self.batch_size)
                if self.data_len % self.batch_size != 0:
                    print(
                        "Warning - training dataset is not a multiple of batch size:",
                        self.batch_size,
                    )
        return smiles, targets
