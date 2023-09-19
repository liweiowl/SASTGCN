import os
import pandas as pd
import numpy as np
import deepdish as dd
from configuration import cf
from torch.utils.data import Dataset
from scipy.spatial.distance import cdist


class ReadDataset(Dataset):
    def __init__(self, x, y):
        self.feature = x
        self.label = y

    def __getitem__(self, idx):
        if self.label is None:
            print(" there is no label")
            return self.feature[idx]
        return self.feature[idx], self.label[idx]

    def __len__(self):
        return len(self.label)


def load_dataset(cf):
    print(f'datasetname is: {cf.dataset_name}')
    print(f"data is stored in:{cf.data_path}")
    data_path = os.path.join(cf.data_path, cf.dataset_name)

    # if cf.dataset_name =="metr-la.h5" or cf.dataset_name == "pems-bay.h5":
    if cf.dataset_name[-3:] == ".h5":
        print('original data is stored in a .h5 file')
        h5f = dd.io.load(data_path)
        keys, values = list(h5f.keys()), list(h5f.values())
        data = values[0].values
        print(data.shape)
    elif cf.dataset_name[-3:] == "npz":
        print('original data is stored in a npz file')
        data = np.load(data_path)
        print(data.shape)
    elif cf.dataset_name[-3:] == 'csv':
        print('original data is stored in a csv file')
        data = pd.read_csv(data_path, header=None).values
        print(data.shape)
    print('data loaded')
    return data


def slide_windows(x, y, seq_len=30, pred_len=3):
    # if y and x are in the same space, then the init y=x
    # x_seq = [x[i:i + seq_len] for i in range(len(x) - seq_len)]
    # y_label = y[seq_len:]
    x_seq = [x[i:i + seq_len] for i in range(len(x) - seq_len - pred_len)]
    y_label = [y[i:i + pred_len] for i in range(seq_len, len(x) - pred_len)]

    return np.array(x_seq), np.array(y_label)


def reverse_slide_windows(x):
    batch, seq_len, num_nodes = x.shape
    x_reverse_slide_windows = x[0, :, :]
    for i in range(1,batch):
        x_reverse_slide_windows = np.vstack((x_reverse_slide_windows, x[i][-1]))
    return x_reverse_slide_windows


class Standardscaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def reverse_transform(self, data):
        return data * self.std + self.mean


class MinMaxscaler():
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def reverse_transform(self, data):
        return (data * (self.max - self.min)) + self.min


def standardscaler_function(data):
    # to scale data which is in the same row
    # print(f"data mean is {np.mean(data, axis=0)}, data std is {np.std(data, axis=0)}")
    # return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # to scale data in all the matrix
    print(f"data mean is {np.mean(data)}, data std is {np.std(data)}")
    return (data - np.mean(data)) / np.std(data)


def cal_adjacent_matrix(data, hidden_size=40, normalized_category="laplacian"):
    # hidden_size = 40
    data = reverse_slide_windows(data)
    # print(" svd decomposition")
    u, s, v = np.linalg.svd(np.array(data))
    # print(" to get the station representation")

    w = np.diag(s[:hidden_size]).dot(v[:hidden_size, :]).T
    # print("calculate the distance between stations")
    graph = cdist(w, w, metric='euclidean')
    # print(" use a Gaussian methond to transfer the distance to weights between stations")
    a = graph * -1 / np.std(graph) ** 2
    support = np.exp(a)
    support = support - np.identity(
        support.shape[0])  # np.identity: creat a matrix (M,M) in which the main diagonal is 1, the rest is 0
    if normalized_category == 'randomwalk':
        support = random_walk_matrix(support)
    elif normalized_category == 'laplacian':
        support = normalized_laplacian(support)
    return support


def random_walk_matrix(w) -> np.matrix:
    d = np.array(w.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = np.eye(d_inv.shape[0]) * d_inv
    return d_mat_inv.dot(w)


def normalized_laplacian(w: np.ndarray) -> np.matrix:
    d = np.array(w.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # print(d, d_inv_sqrt)
    d_mat_inv_sqrt = np.eye(d_inv_sqrt.shape[0]) * d_inv_sqrt.shape
    return np.identity(w.shape[0]) - d_mat_inv_sqrt.dot(w).dot(d_mat_inv_sqrt)


if __name__ == "__main__":
    data = load_dataset(cf)
    print("love world")
