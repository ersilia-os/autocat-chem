import csv
import numpy as np
import random


class DataReader(object):
    def __init__(self, input_file):
        self.file_name = input_file
        self.len = self.get_num_lines()

    def read_length(self):
        return self.len

    def get_num_lines(self):
        with open(self.file_name) as f:
            csv_reader = csv.reader(f, delimiter=",")
            header = next(csv_reader)
            sum = 0
            for i in csv_reader:
                sum += 1

            return sum

    def read_all(self):
        return self.get_fold(0, self.len)

    def get_fold(self, fold_num, batch_size):
        smiles, target = [], []

        with open(self.file_name) as f:
            csv_reader = csv.reader(f, delimiter=",")
            if fold_num == 0:
                header = next(csv_reader)

            low = fold_num * batch_size
            high = (fold_num + 1) * batch_size
            for i, row in enumerate(csv_reader):
                if i >= low and i < high:
                    smiles.append(row[0])
                    float_row = [float(num) for num in row[1:]]
                    target.append(float_row)

        return np.array(smiles), np.array(target, dtype=np.float32)

    def random_fold(self):
        num_folds = self.len // self.batch_size
        fold = random() * num_folds
        return self.get_fold(fold, self.batch_size)

    def read_smiles(self, smiles_col=0):
        smiles = []

        with open(self.file_name) as f:
            csv_reader = csv.reader(f, delimiter=",")
            header = next(csv_reader)
            for row in csv_reader:
                smiles.append(row[smiles_col])

        return smiles
