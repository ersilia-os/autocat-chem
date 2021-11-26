from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import catboost
from catboost import Pool, CatBoostRegressor
import json
import sys
import os

from .base import AutoCatTrain
from .param_optimizer import Optimizer

BATCH_ITERATIONS = 3
HIST_BINS = 100

class AutoCatFitter(AutoCatTrain):
    def __init__(self, smiles, targets, scaler, features_file=None, batch=False, data_r=None, batch_size=0, data_len=0):
        AutoCatTrain.__init__(self)
        self.scaler = scaler
        self.features_file = features_file
        self.X = self.featurizer.featurize(smiles, features_file=self.features_file)
        self.y = self.scaler.scale_data(targets)

        self.do_batching = batch
        self.batch_size = batch_size
        self.data_len = data_len
        self.data_r = data_r
        self.batch_iter = BATCH_ITERATIONS

        self.weighting = False

        self.train_params(self.y)
        self.optuna_params = {}

    def weight_labels(self):
        self.hist_weights = np.zeros((self.y.shape[1], HIST_BINS))
        self.hist_bins = np.zeros((self.y.shape[1], HIST_BINS+1))

        for col in range(self.y.shape[1]):
            weights_col, bin_col = np.histogram(self.y[:,col], bins=HIST_BINS, density=True)
            minmaxscaler = MinMaxScaler()
            weights_scaled = minmaxscaler.fit_transform(weights_col.reshape((HIST_BINS, 1)))
            self.hist_weights[col] = weights_scaled.reshape((HIST_BINS,))
            self.hist_bins[col] = bin_col
        self.weighting = True

    def optimise_search(self, time_budget=3600):
        if self.weighting:
            self.opt = Optimizer(self.X, self.y, hist_weights=self.hist_weights, bins=self.hist_bins)
        else:
            self.opt = Optimizer(self.X, self.y)
        self.optuna_params = self.opt.param_search(time_budget)

    def fit(self):
        if self.do_batching:
            self.fit_model(self.X, self.y)
            self.save_model("temp.cbm", "cbm")
            print("Training final model on fold 0 and iteration 0.")

            fold = 1
            for i in range(self.batch_iter):
                for f in range(fold, self.data_len // self.batch_size):
                    smiles, targets = self.data_r.get_fold(f, self.batch_size)
                    del self.X
                    del self.y
                    self.X = self.featurizer.featurize(smiles, features_file=self.features_file)
                    self.y = self.scaler.scale_data(targets)

                    print("Training final model on fold", int(f), "and iteration", int(i))
                    self.fit_model(self.X, self.y, init_model="temp.cbm")
                    self.save_model("temp.cbm", "cbm")

                if i != self.batch_iter-1:
                    self.save_model("chkpt_iteration_" + str(i) + ".cbm", "cbm")
                    self.save_metrics("chkpt_iteration" + str(i) + "_metrics.json")
                    self.scaler.save("chkpt_iteration" + str(i) + "_scaler.json")
                fold = 0
            os.remove("temp.cbm")
            return self.metrics

        else:
            print("Training final model.")
            return self.fit_model(self.X, self.y)

    def fit_model(self, X, y, init_model=None, log_path=""):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        if self.weighting:
            weights_y_train = self.get_weights(y_train, self.hist_weights, self.hist_bins)
            weights_y_test = self.get_weights(y_test, self.hist_weights, self.hist_bins)
            dtrain = Pool(X_train, label=y_train, weight=weights_y_train)
            dtest = Pool(X_test, label=y_test, weight=weights_y_test)
        else:
            dtrain = Pool(X_train, label=y_train)
            dtest = Pool(X_test, label=y_test)

        params = self.training_params
        if len(self.optuna_params) > 0:
            params.update(self.optuna_params)

        if log_path == "":
            log_path = "catboost_training" + ".log"

        file_mode = "w"
        if init_model is not None:
            params['task_type'] = "CPU"
            file_mode = "a"

        self.model = CatBoostRegressor(**params)
        with open(log_path, file_mode) as f:
            self.model.fit(dtrain, eval_set=dtest, early_stopping_rounds=100, log_cout=f, init_model=init_model)

        self.model_metrics(X_train, X_test, y_train, y_test)
        return self.metrics

    def model_metrics(self, X_train, X_test, y_train, y_test):
        preds_train = self.model.predict(X_train)
        preds_test = self.model.predict(X_test)

        self.metrics = {
            "MAE_train":  mean_squared_error(y_train, preds_train),
            "MAE_test": mean_squared_error(y_test, preds_test),
            "r2_train": r2_score(y_train, preds_train),
            "r2_test": r2_score(y_test, preds_test)
        }

    def save_model(self, file_path, format):
        if format == "onnx":
            self.model.save_model(file_path, format=format, export_parameters={
                'onnx_domain': 'ai.catboost',
                'onnx_model_version': 1,
                'onnx_doc_string': 'Model for Regression',
                'onnx_graph_name': 'CatBoostModel_for_Regression'
            })
        else:
            self.model.save_model(file_path, format=format)

    def save_metrics(self, file_path):
        output = self.metrics
        versions = {
            "catboost_version": catboost.__version__,
            "rdkit_version": self.featurizer.rdkit_version(),
            "python_version": str(sys.version_info[0]) + "." + str(sys.version_info[1])
        }

        output.update({"versions": versions})
        with open(file_path, "w") as f:
            json.dump(output, f)

    def save_weights(self, file_path):
        if self.weighting:
            output = {
                "histogram_bins": self.hist_bins.tolist(),
                "histogram_weights": self.hist_weights.tolist()
            }
            with open(file_path, "w") as f:
                json.dump(output, f)

    def get_model(self):
        return self.model
