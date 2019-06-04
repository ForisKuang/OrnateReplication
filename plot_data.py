import matplotlib.pyplot as plt
import pickle
from main import get_file_list

pkl_files = get_file_list('/net/scratch/aivan/decoys/ornate/pkl.rand70', 20)
dim0 = []
dim1 = []
dim2 = []
dim3 = []
dim4 = []
scores = []
num_bins = 50

for pkl_file in pkl_files:
    with open(pkl_file, 'rb') as f:
        protein = pickle.load(f, encoding='latin1')
        features = protein['features']
        protein_scores = protein['scores']
        for i in range(len(features)):
            dim0.extend(features[i][0])
            dim1.extend(features[i][1])
            dim2.extend(features[i][2])
            dim3.extend(features[i][3])
            dim4.extend(features[i][4])
        scores.extend(protein_scores)

# Create histograms
print('Dim0 Max', max(dim0), 'min', min(dim0))
print('Dim1 Max', max(dim1), 'min', min(dim1))
print('Dim2 Max', max(dim2), 'min', min(dim2))
print('Dim3 Max', max(dim3), 'min', min(dim3))
print('Dim4 Max', max(dim4), 'min', min(dim4))

n, bins, patches = plt.hist(dim0, num_bins, facecolor='blue', alpha=0.5)
plt.savefig('dim0_hist.png')
plt.clf()
plt.n, bins, patches = plt.hist(dim1, num_bins, facecolor='blue', alpha=0.5)
plt.savefig('dim1_hist.png')
plt.clf()
n, bins, patches = plt.hist(dim2, num_bins, facecolor='blue', alpha=0.5)
plt.savefig('dim2_hist.png')
plt.clf()
n, bins, patches = plt.hist(dim3, num_bins, facecolor='blue', alpha=0.5)
plt.savefig('dim3_hist.png')
plt.clf()
n, bins, patches = plt.hist(dim4, num_bins, facecolor='blue', alpha=0.5)
plt.savefig('dim4_hist.png')
plt.clf()
n, bins, patches = plt.hist(scores, num_bins, facecolor='blue', alpha=0.5)
plt.savefig('scores_hist.png')
plt.clf()
