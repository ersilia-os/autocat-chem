import numpy as np
import json

CLIP_BOUNDS = [-10, 10]

class Scaler(object):
    def __init__(self):
        self.means = []
        self.std_devs = []
        self.clip_bounds = CLIP_BOUNDS

    def get_params(self, data_arr):
        self.means = np.nanmean(data_arr, axis=0)
        self.std_devs = np.nanstd(data_arr, axis=0)

    def scale_data(self, data_arr):
        means = np.nanmean(data_arr, axis=0)
        for col in range(data_arr.shape[1]):
            for row in range(data_arr.shape[0]):
                if np.isnan(data_arr[row,col]):
                    data_arr[row,col] = means[col]
                data_arr[row,col] = (data_arr[row,col] - self.means[col]) / self.std_devs[col]
                data_arr[row,col] = self.clip(data_arr[row,col])
        return data_arr

    def reverse_scale(self, data_arr):
        for col in range(data_arr.shape[1]):
            for row in range(data_arr.shape[0]):
                data_arr[row,col] = data_arr[row,col] * self.std_devs[col] + self.means[col]
        return data_arr

    def clip(self, num):
        if num < self.clip_bounds[0]:
            return self.clip_bounds[0]
        elif num > self.clip_bounds[1]:
            return self.clip_bounds[1]
        return num

    def save(self, file_path):
        json_data = {
            "means": self.means.tolist(),
            "std_devs": self.std_devs.tolist()
        }
        with open(file_path, "w") as f:
            json.dump(json_data, f)

    def load(self, file_path):
        with open(file_path, "r") as f:
            json_data = json.load(f)
        self.means = np.array(json_data["means"])
        self.std_devs = np.array(json_data["std_devs"])
