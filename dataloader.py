import numpy as np
from torch.utils.data import Dataset
import scipy.io #处理mat数据

class Caltech101_20(Dataset):
    def __init__(self, path):
        self.y = scipy.io.loadmat(path + 'Caltech101_20.mat')['Y'].astype(np.int32).reshape(2386,)
        self.V1 = scipy.io.loadmat(path + 'Caltech101_20.mat')['X'][0][0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Caltech101_20.mat')['X'][0][1].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Caltech101_20.mat')['X'][0][2].astype(np.float32)
        self.V4 = scipy.io.loadmat(path + 'Caltech101_20.mat')['X'][0][3].astype(np.float32)
        self.V5 = scipy.io.loadmat(path + 'Caltech101_20.mat')['X'][0][4].astype(np.float32)
        self.V6 = scipy.io.loadmat(path + 'Caltech101_20.mat')['X'][0][5].astype(np.float32)

        self.data1 = self.V1.reshape(2386, 48)
        self.data2 = self.V2.reshape(2386, 40)
        self.data3 = self.V3.reshape(2386, 254)
        self.data4 = self.V4.reshape(2386, 1984)
        self.data5 = self.V5.reshape(2386, 512)
        self.data6 = self.V6.reshape(2386, 928)

    def __len__(self):
        return 2386

def load_data(dataset):
    if dataset == "Caltech101_20":
        dataset = Caltech101_20('./datasets/')
        dims = [48, 40, 254, 1984, 512, 928]
        view = 5
        data_size = 2386
        class_num = 20
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num

