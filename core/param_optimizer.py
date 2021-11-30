import optuna
from optuna.samplers import CmaEsSampler, TPESampler, MOTPESampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import Pool, CatBoostRegressor
from time import perf_counter

from .base import AutoCatTrain
from .defaults import MAX_TREE_DEPTH


class Optimizer(AutoCatTrain):
    def __init__(self, X, y, hist_weights=None, bins=None, reference_lib=None):
        AutoCatTrain.__init__(self)
        self.X = X
        self.y = y
        self.hist_weights = hist_weights
        self.bins = bins
        self.train_params(self.y)
        self.reference_lib = reference_lib

    def param_search(self, time_budget=3600):
        print("Starting hyperparameter time trial for 20s")
        t1 = perf_counter()
        study_time_check = optuna.create_study(
            sampler=TPESampler(), direction="minimize"
        )
        study_time_check.optimize(self.objective, n_trials=1, timeout=20)
        t2 = perf_counter()

        print("Starting hyperparameter search for", time_budget, "seconds.")
        if self.y.shape[1] > 1:
            self.study = optuna.create_study(
                sampler=MOTPESampler(), direction="minimize"
            )  # Multiobjective sampler
        elif time_budget / (t2 - t1) > 150:
            self.study = optuna.create_study(
                sampler=CmaEsSampler(), direction="minimize"
            )
        else:
            self.study = optuna.create_study(sampler=TPESampler(), direction="minimize")

        self.study.optimize(self.objective, n_trials=500, timeout=time_budget)
        self.trial = self.study.best_trial.params
        print("Best trial parameters:", self.trial)
        return self.trial

    def objective(self, trial):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2
        )
        if self.hist_weights is not None:
            weights_y_train = self.get_weights(y_train, self.hist_weights, self.bins)
            weights_y_test = self.get_weights(y_test, self.hist_weights, self.bins)
            dtrain = Pool(X_train, label=y_train, weight=weights_y_train)
            dtest = Pool(X_test, label=y_test, weight=weights_y_test)
        else:
            dtrain = Pool(X_train, label=y_train)
            dtest = Pool(X_test, label=y_test)
        trial_params = self.training_params

        tree_depth = 7
        if self.device == "GPU":
            trial_params.update(
                {
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 20),
                    "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-2, 1e0),
                    "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 20),
                }
            )
            if self.reference_lib is None:
                tree_depth = MAX_TREE_DEPTH

        else:
            trial_params.update({"iterations": 100})
            tree_depth = MAX_TREE_DEPTH

        trial_params.update(
            {
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-1, 1e0),
                "depth": trial.suggest_int("depth", 4, tree_depth),
            }
        )

        reg = CatBoostRegressor(**trial_params)
        reg.fit(dtrain, eval_set=dtest, early_stopping_rounds=100, verbose=0)
        y_pred = reg.predict(X_test)
        score = mean_absolute_error(y_test, y_pred)
        return score

    def view_param_importance(self):
        optuna.visualization.plot_param_importances(self.study)
