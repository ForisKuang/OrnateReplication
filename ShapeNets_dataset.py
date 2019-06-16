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
    def __init__(self, filename, label, lower_bound=None, upper_bound=None):
        """
        Initializes a dataset for the given pickle file.
        
        Each element represents a residue and consists of a sparse representation
        of a 4-D tensor. Conceptually, at each 3-D grid location, there is a
        vector of ~167 "atom densities" (TODO - is this the right terminology). 
        To save memory, we only store non-zero entries. Each row contains
        (x, y, z, atom #, value), where the first 4 values represent the
        indexes, and the last value is the actual feature value.

        In addition, the residue is labeled with its quality score, UNLESS
        the "label" parameter is set, in that "label" gets applied to all
        points in the dataset.

        If upper_bound and/or lower_bound are set, we only include residues
        whose scores fall within that range.
        """
        self.data = np.load(filename)  # 3D Shape Nets
        print('Data shape', self.data.shape)
        self.labels = [label] * data.shape[0]  # Labels (quality score)
        
    def __len__(self):
        return 1

    def __getitem__(self, idx): 
        return {'inputs': torch.FloatTensor(self.data),
                'labels': torch.FloatTensor([self.labels[idx]]) }

