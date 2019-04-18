import glob
import os
import pickle

import numpy as np
import torch
import torch.utils.data as _data
from sklearn.utils import shuffle

def load_dataset(pickle_dir, num_files=None, starting_index=0, fake=False):
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
    THRESHOLD = 0.4

    # Find all .pkl files within pickle_dir (including subdirectories)
    file_list = glob.glob(pickle_dir + '/**/*.pkl', recursive=True)
    if num_files:
        file_list = file_list[0:num_files]
    for pickle_file in file_list:
        path = os.path.join(pickle_dir, pickle_file)
        print(path)
        with open(path, 'rb') as f:
            protein = pickle.load(f, encoding='latin1')
            features = protein['features']
            labels = protein['scores']
            print('len features ' + str(len(features)))
            for i in range(len(features)):
                # If we're loading modeled structures, exclude the ones that are "realistic" (score > 0.6)
                if fake:
                    print(labels[i])
                    if labels[i] > THRESHOLD:
                        continue
                idx2map2d[idx] = features[i]
                X.append(idx)
                idx += 1
                Y.append(labels[i])

    print('Returning ' + str(len(Y)) + ' examples.')
    return X, Y, idx2map2d
