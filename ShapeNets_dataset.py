import glob
import os
import pickle
import numpy as np
import random
import torch
import torch.utils.data as _data
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle


class ShapeNetsDataset(Dataset):
    def __init__(self, real_filename, fake_filename):
        """
        Initializes an entry in the 3D shapes dataset. This simultaneously reads
        in a real 3D shape and the corresponding fake shape from the  (the mapping is important here).
        """
        self.real_data = np.load(real_filename)
        self.fake_data = np.load(fake_filename)
        
        # To match the format of the protein dataset, add a 5th dimension
        # (the features at each pixel - even though there's only 1 feature)
        self.real_data = np.expand_dims(self.real_data, 4)
        self.fake_data = np.expand_dims(self.fake_data, 4)

    def __len__(self):
        return 1

    def __getitem__(self, idx): 
        return {'real_data': torch.FloatTensor(self.real_data),
                'fake_data': torch.FloatTensor(self.fake_data)}
