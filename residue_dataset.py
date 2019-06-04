import glob
import os
import pickle
import numpy as np
import random
import torch
import torch.utils.data as _data
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle


class ResidueDataset(Dataset):
    def __init__(self, pkl_file, lower_bound=None, upper_bound=None, label=None):
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
        self.data = []  # Residues
        self.labels = []  # Labels (quality score)
        
        with open(pkl_file.strip(), 'rb') as f:
            protein = pickle.load(f, encoding='latin1')
            features = protein['features']
            scores = protein['scores']
            for i in range(len(features)):
                if lower_bound:
                    if scores[i] < lower_bound:
                        continue
                if upper_bound:
                    if scores[i] > upper_bound:
                        continue
                residue_map = np.transpose(features[i])
                self.data.append(residue_map)
                if label is not None:
                    self.labels.append(label)
                else:
                    self.labels.append(scores[i])          

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        map2d = self.data[idx]
        map4d = np.zeros((24, 24, 24, 167))
        map4d[map2d[:,0],map2d[:,1],map2d[:,2],map2d[:,3]] = map2d[:,4]
        return {'inputs': torch.FloatTensor(map4d),
                'labels': torch.FloatTensor([self.labels[idx]]) }

