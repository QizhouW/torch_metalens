from torchvision import datasets, transforms
from base import BaseDataLoader
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from skimage.transform import resize
from skimage.transform import AffineTransform, warp, rotate
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import sys
import pandas as pd
import torchvision

def random_shift(image, range=20):
    x_shift, y_shift = np.random.randint(low=-range // 2, high=range // 2, size=2)
    transform = AffineTransform(translation=[x_shift, y_shift])
    shifted = warp(image, transform, mode='wrap', preserve_range=True)
    shifted = shifted.astype(image.dtype)
    return shifted


def unwrap_phase(p):
    p = np.unwrap(p)
    if p[0] > (2 * np.pi):
        p = p - 2 * np.pi
    return p


class MyDataSet(Dataset):
    def __init__(self, data_dir, csv_file,
                 thickness, polarization, unit_size=50, rotate_augmentation=True,
                 overlap=True, datatype='phase', train_test='train',
                 shift_range=20, taget_normalize=6.2832):

        self.dim = unit_size
        self.thickness = thickness
        self.polarization = polarization
        self.shift_range=shift_range
        self.data_list = pd.read_csv(os.path.join(data_dir, csv_file), index_col=0)
        assert self.dim == np.unique(self.data_list.nx), 'The input unit size is wrong'
        X = np.load(os.path.join(data_dir, 'geo.npy'))
        assert X.dtype == 'uint8', 'Image dtype is not uint8'

        if not overlap:
            self.data_list = self.data_list[self.data_list.overlap == False]
        self.data_list = self.data_list[self.data_list.dz == thickness]
        self.data_list = self.data_list[self.data_list.status == train_test]
        data_idx = self.data_list.index
        X = X[data_idx]

        if datatype == 'phase':
            Y = np.load(os.path.join(data_dir, 'phase.npy'))
            Y = Y[data_idx]
            Y_Ex = [unwrap_phase(d[0]) / taget_normalize for d in Y]
            Y_Ey = [unwrap_phase(d[1]) / taget_normalize for d in Y]
        elif datatype == 'amplitude':
            Y = np.load(os.path.join(data_dir, 'amp.npy'))
            Y = Y[data_idx]
            Y_Ex = [d[0] for d in Y]
            Y_Ey = [d[1] for d in Y]

        if self.polarization == 'Ex':
            Y = Y_Ex
            if rotate_augmentation:
                self.X = np.concatenate([X, np.transpose(X, (0, 2, 1))], axis=0)
                self.Y = np.array(Y + Y_Ey)
        elif self.polarization == 'Ey':
            Y = Y_Ey
            if rotate_augmentation:
                self.X = np.concatenate([X, np.transpose(X, (0, 2, 1))], axis=0)
                self.Y = np.array(Y + Y_Ex)

        if shift_range:
            self.transform = random_shift
        else:
            self.transform = None

    def __len__(self):
        assert len(self.X)==len(self.Y), 'parsing wrong labels'
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.Y[idx]
        if self.transform:
            image = self.transform(image,self.shift_range)
        image= torchvision.transforms.functional.to_tensor(image)

        return image, label.astype('float32')


class MyDataGen(BaseDataLoader):
    """
    METALENS data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, csv_file,
                 thickness, polarization, unit_size=50, rotate_augmentation=True,
                 overlap=True, datatype='phase', train_test='train',
                 shift_range=20, taget_normalize=6.2832,
                 batch_size=32, shuffle=True, validation_split=0.1,
                 num_workers=1):

        self.dataset =  MyDataSet(data_dir, csv_file,
                 thickness, polarization, unit_size, rotate_augmentation,
                 overlap, datatype, train_test,
                 shift_range, taget_normalize)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
