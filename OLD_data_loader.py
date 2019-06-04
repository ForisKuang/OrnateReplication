import glob
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.utils.data as _data
from sklearn.utils import shuffle

def load_dataset(pickle_dir, num_files=None, starting_index=0, lower_bound=None, upper_bound=None):
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

    # For plotting hist
    dim0 = []
    dim1 = []
    dim2 = []
    dim3 = []
    dim4 = []
    num_bins=50


    # Find all .pkl files within pickle_dir (including subdirectories)
    file_list = glob.glob(pickle_dir + '/**/*.pkl', recursive=True)
    random.shuffle(file_list)
    if num_files:
        file_list = file_list[0:num_files]
    values = []

    for pickle_file in file_list:
        path = os.path.join(pickle_dir, pickle_file)
        print(path)
        with open(path, 'rb') as f:
            protein = pickle.load(f, encoding='latin1')
            features = protein['features']
            labels = protein['scores']
            for i in range(len(features)):
                if lower_bound:
                    if labels[i] < lower_bound:
                        continue
                if upper_bound:
                    if labels[i] > upper_bound:
                        continue
                idx2map2d[idx] = features[i]
                X.append(idx)
                idx += 1
                Y.append(labels[i])
                #dim0.extend(features[i][0])
                #dim1.extend(features[i][1])
                #dim2.extend(features[i][2])
                #dim3.extend(features[i][3])
                #dim4.extend(features[i][4])

    #print('Dim0 Max', max(dim0), 'min', min(dim0))
    #print('Dim1 Max', max(dim1), 'min', min(dim1))
    #print('Dim2 Max', max(dim2), 'min', min(dim2))
    #print('Dim3 Max', max(dim3), 'min', min(dim3))
    #print('Dim4 Max', max(dim4), 'min', min(dim4))
    #n, bins, patches = plt.hist(dim0, num_bins, facecolor='blue', alpha=0.5)
    #plt.savefig('dim0_hist.png')
    #n, bins, patches = plt.hist(dim1, num_bins, facecolor='blue', alpha=0.5)
    #plt.savefig('dim1_hist.png')
    #n, bins, patches = plt.hist(dim2, num_bins, facecolor='blue', alpha=0.5)
    #plt.savefig('dim2_hist.png')
    #n, bins, patches = plt.hist(dim3, num_bins, facecolor='blue', alpha=0.5)
    #plt.savefig('dim3_hist.png')
    #n, bins, patches = plt.hist(dim4, num_bins, facecolor='blue', alpha=0.5)
    #plt.savefig('dim4_hist.png')

    print('Returning ' + str(len(Y)) + ' examples.')
    return X, Y, idx2map2d
