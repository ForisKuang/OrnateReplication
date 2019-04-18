import glob
import os
import pickle

import numpy as np
import torch
import torch.utils.data as _data
from sklearn.utils import shuffle

def load_dataset(pickle_dir, starting_index=0):
    """
    Constructs a dataset from all the pickle files inside "pickle_dir"
    (including subdirectories). Note that the dataset is NOT shuffled.
    The dataset is formatted as:

    X: a 1-D list of indexes (one index per residue). These are numbered
       starting from "starting_index."
    Y: a 1-D list of labels (one per residue), in the same order of the indexes
    idx2map2d: maps from the residue index to the 2-D representation of its structure.
               In that 2-D representation, each row represents a non-zero entry in
               the 4-D space; the first 3 indexes give the 3D coordinates, the next
               index gives the feature number, and the final index contains the value.
    """
    idx2map2d = {}
    idx = starting_index
    X = []
    Y = []

    # Adjust number of files to read
    NUM_FILES = 1

    # Find all .pkl files within pickle_dir (including subdirectories)
    for pickle_file in glob.glob(pickle_dir + '/**/*.pkl', recursive=True)[0:NUM_FILES]:
        path = os.path.join(pickle_dir, pickle_file)
        print(path)
        with open(path, 'rb') as f:
            protein = pickle.load(f, encoding='latin1')
            features = protein['features']
            labels = protein['scores']
            for map2d in features:
                idx2map2d[idx] = map2d
                X.append(idx)
                idx += 1
            for label in labels:
                Y.append(label)
    return X, Y, idx2map2d
