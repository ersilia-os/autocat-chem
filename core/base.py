import GPUtil
from .featurizer import Featurizer
from .scaler import Scaler
from .defaults import TRAIN_TEST_SPLIT, TRY_GPU, HIST_BINS
import numpy as np


class AutoCatBase(object):
    def __init__(self):
        self.featurizer = Featurizer()
        self.scaler = Scaler()


class AutoCatTrain(AutoCatBase):
    def __init__(self):
        AutoCatBase.__init__(self)
        self.train_test_split = TRAIN_TEST_SPLIT
        self.device = self.find_GPUs()

    def find_GPUs(self):
        GPUs = GPUtil.getGPUs()
        if len(GPUs) > 0 and TRY_GPU:
            return "GPU"
        return "CPU"

    def train_params(self, labels):
        training_params = {}
        if labels.shape[1] > 1:
            training_params["loss_function"] = "MultiRMSE"
            training_params["eval_metric"] = "MultiRMSE"
            training_params["task_type"] = "CPU"
        else:
            training_params["loss_function"] = "RMSE"
            training_params["eval_metric"] = "RMSE"

        training_params["task_type"] = self.device
        return training_params

    def get_weights(self, data, hist_weights, bins, bin_range=[0, HIST_BINS - 1]):
        weights = np.zeros((data.shape[1], data.shape[0]))
        for col in range(data.shape[1]):
            col_indxs = np.digitize(data[:, col], bins[col])
            for i, val in enumerate(col_indxs):
                if val < bin_range[0]:
                    col_indxs[i] = bin_range[0]
                elif val > bin_range[1]:
                    col_indxs[i] = bin_range[1]

            col_weights = [
                self.weight_function(hist_weights[col, i], i) for i in col_indxs
            ]
            weights[col] = np.array(col_weights).ravel()
        weights_avg = np.nanmean(weights, axis=0)
        return weights_avg

    def weight_function(self, bin_weight, bin_pos):
        return 1 - bin_weight
