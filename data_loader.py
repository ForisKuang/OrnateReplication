import os
import pickle

import numpy as np
import torch
import torch.utils.data as _data
from sklearn.utils import shuffle

def load_dataset(pickle_dir):
    idx2map2d = {}
    idx = 0
    Y = []
    X = []
    for _, _, files, in os.walk(pickle_dir):
        for file in files:
            if file.endswith(".pkl"):
                path = os.path.join(pickle_dir, file)
                with open(path, 'rb') as f:
                    protein = pickle.load(f)
                    features = protein['features']
                    labels = protein['scores']
                    for map2d in features:
                        idx2map2d[idx] = map2d
                        X.append(idx)
                        idx += 1
                    for label in labels:
                        Y.append(label)
        X, Y = shuffle(X, Y)
        return X, Y, idx2map2d
