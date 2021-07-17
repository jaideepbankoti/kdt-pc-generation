import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from utils.helpers import h_read
from pathlib import Path
from sklearn.utils.extmath import randomized_svd
import pdb

class ToTensor(object):
    def __call__(self, data):
        return torch.from_numpy(data.astype(np.float32))

def default_transforms():
    return transforms.Compose([
        ToTensor(),
    ])

class PCA_optim(Dataset):
    """Loads preprocessed points dataset"""

    def __init__(self, data_dir='assets/process_data', transform=default_transforms(), val=False, vslice=None, shape_basis=100):
        r"""
        Args:
            data_dir (string): Path to the preprocessed dataset
            transform (opt): optional transforms to be applied
            val(opt): for the validation set
            vslice(opt): helpful for splitting into test and train slices
            shape_basis: linear shape basis value
        """
        self.transform = transform
        dataset = h_read(Path(data_dir) / 'dataset.h5', 'data')
        U, S, Vt = randomized_svd(dataset, shape_basis)
        self.principal_c = np.matmul(U, np.diag(S))
        self.data = Vt
        cutoff = int(0.8 * self.data.shape[1])     # 80% for training
        # pdb.set_trace()
        if val  == False:
            RAND_SLICES_TRAIN = np.random.choice(self.data.shape[1], cutoff)
            self.vslice = np.delete(np.arange(self.data.shape[1]), RAND_SLICES_TRAIN)
        if val:
            self.data = self.data[:, vslice]
        else:
            self.data = self.data[:, RAND_SLICES_TRAIN]

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        pdata = self.principal_c                        # (3000, 100)
        idata = self.data[:, idx]                       # (100, )
        if self.transform:
            pdata = self.transform(pdata)
            idata = self.transform(idata)
        return pdata, idata

class ToTensor(object):
    def __call__(self, data):
        return torch.from_numpy(data)

