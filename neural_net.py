import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 1, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(72 * 72, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class HDF5DataSet(Dataset):

    def __init__(self, hdf5_dir, transform=None, target_transform=None):
        self.hdf5_dir = h5py.File(hdf5_dir, 'r')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.hdf5_dir.keys())

    def __getitem__(self, index):
        group = self.hdf5_dir[str(index)]
        image = np.array(group['image'])
        label = np.array(group['label'])

        return torch.tensor(image).float(), label[0]