from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import catboost
from catboost import Pool, CatBoostRegressor, sum_models
from .defaults import BATCH_ITERATIONS, HIST_BINS, LOG_PATH
import json
import sys
import os
import random

from .base import AutoCatTrain
from .param_optimizer import Optimizer


class AutoCatFitter(AutoCatTrain):
    def __init__(
        self,
        scaler,
        features_file=None,
        batch=False,
        data_r=None,
        batch_size=0,
        data_len=0,
        training_params={},
    ):
        AutoCatTrain.__init__(self)
        self.scaler = scaler
        self.features_file = features_file

        self.do_batching = batch
        self.batch_size = batch_size
        self.data_len = data_len
        self.data_r = data_r
        self.batch_iter = BATCH_ITERATIONS

        self.weighting = False

        self.training_params = training_params
        self.optuna_params = {}

    def weight_labels(self, targets):
        y = self.scaler.scale_data(targets)
        self.hist_weights = np.zeros((y.shape[1], HIST_BINS))
        self.hist_bins = np.zeros((y.shape[1], HIST_BINS + 1))

        for col in range(y.shape[1]):
            weights_col, bin_col = np.histogram(y[:, col], bins=HIST_BINS, density=True)
            minmaxscaler = MinMaxScaler()
            weights_scaled = minmaxscaler.fit_transform(
                weights_col.reshape((HIST_BINS, 1))
            )
            self.hist_weights[col] = weights_scaled.reshape((HIST_BINS,))
            self.hist_bins[col] = bin_col
        self.weighting = True

    def optimise_search(self, smiles, targets, time_budget=3600):
        X = self.featurizer.featurize(smiles, features_file=self.features_file)
        y = self.scaler.scale_data(targets)

        if self.weighting:
            self.opt = Optimizer(
                X,
                y,
                hist_weights=self.hist_weights,
                bins=self.hist_bins,
                reference_lib=self.features_file,
                featurizer=self.featurizer,
            )
        else:
            self.opt = Optimizer(
                X, y, reference_lib=self.features_file, featurizer=self.featurizer
            )
        self.optuna_params = self.opt.param_search(time_budget)

    def fit(self, smiles_inp, targets_inp, retrain=None):
        with open(LOG_PATH, "w") as f:
            pass  # Flush log file for new training run

        if self.do_batching:
            for i in range(self.batch_iter):
                seed = random.randint(0, 10000)
                models = []
                for f in range(self.data_len // self.batch_size):
                    print(
                        "Training final model on fold", int(f), "and iteration", int(i)
                    )
                    smiles, targets = self.data_r.get_fold(f, self.batch_size)
                    X = self.featurizer.featurize(
                        smiles, features_file=self.features_file
                    )
                    y = self.scaler.scale_data(targets)

                    if i == 0:
                        init_model = retrain
                        models.append(
                            self.fit_model(X, y, init_model=init_model, seed=seed)
                        )

                    elif i > 0:
                        init_model = "temp.cbm"
                        self.model = self.fit_model(
                            X, y, init_model=init_model, seed=seed
                        )
                        self.save_model("temp.cbm", "cbm")

                if i == 0:
                    model_avg = sum_models(
                        models, weights=[1.0 / len(models)] * len(models)
                    )
                    self.model = model_avg
                    del models
                    self.save_model("temp.cbm", "cbm")

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=seed
                )
                self.metrics = self.model_metrics(
                    model_avg, X_train, X_test, y_train, y_test
                )

                if i != self.batch_iter - 1:
                    self.save_model("chkpt_iteration_" + str(i) + ".cbm", "cbm")
                    self.save_metrics("chkpt_iteration" + str(i) + "_metrics.json")
                    self.scaler.save("chkpt_iteration" + str(i) + "_scaler.json")

            os.remove("temp.cbm")
            return self.metrics

        else:
            print("Training final model.")
            X = self.featurizer.featurize(smiles_inp, features_file=self.features_file)
            y = self.scaler.scale_data(targets_inp)
            seed = random.randint(0, 10000)

            self.model = self.fit_model(X, y, seed=seed)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )
            self.metrics = self.model_metrics(
                self.model, X_train, X_test, y_train, y_test
            )

    def fit_model(
        self, X, y, init_model=None, log_path="", seed=random.randint(0, 10000)
    ):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        if self.weighting:
            weights_y_train = self.get_weights(
                y_train, self.hist_weights, self.hist_bins
            )
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
            log_path = LOG_PATH

        if init_model is not None:
            params["task_type"] = "CPU"

        model = CatBoostRegressor(**params)
        with open(log_path, "a") as f:
            model.fit(
                dtrain,
                eval_set=dtest,
                early_stopping_rounds=100,
                log_cout=f,
                init_model=init_model,
            )
        return model

    def model_metrics(self, model, X_train, X_test, y_train, y_test):
        preds_train = model.predict(X_train)
        preds_test = model.predict(X_test)

        metrics = {
            "MAE_train": mean_squared_error(y_train, preds_train),
            "MAE_test": mean_squared_error(y_test, preds_test),
            "r2_train": r2_score(y_train, preds_train),
            "r2_test": r2_score(y_test, preds_test),
        }
        return metrics

    def save_model(
        self, file_path, format
    ):  # TO DO: Save & load training params. Change learning rate for retrain?
        if format == "onnx":
            self.model.save_model(
                file_path,
                format=format,
                export_parameters={
                    "onnx_domain": "ai.catboost",
                    "onnx_model_version": 1,
                    "onnx_doc_string": "Model for Regression",
                    "onnx_graph_name": "CatBoostModel_for_Regression",
                },
            )
        else:
            self.model.save_model(file_path, format=format)

    def save_metrics(self, file_path):
        output = self.metrics
        versions = {
            "catboost_version": catboost.__version__,
            "rdkit_version": self.featurizer.rdkit_version(),
            "python_version": str(sys.version_info[0]) + "." + str(sys.version_info[1]),
        }

        output.update({"versions": versions})
        with open(file_path, "w") as f:
            json.dump(output, f)

    def save_weights(self, file_path):
        if self.weighting:
            output = {
                "histogram_bins": self.hist_bins.tolist(),
                "histogram_weights": self.hist_weights.tolist(),
            }
            with open(file_path, "w") as f:
                json.dump(output, f)

    def load_weights(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                json_data = json.load(f)
            self.hist_weights = np.array(
                json_data["histogram_weights"], dtype=np.float32
            )
            self.hist_bins = np.array(json_data["histogram_bins"], dtype=np.float32)
            self.weighting = True

    def get_model(self):
        return self.model
