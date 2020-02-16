import numpy as np
import matplotlib as plt
import pickle
from ivector_engine import utils
from sklearn.model_selection import KFold

train_labels = utils.text_file_to_np_array('../docs/train_labels.txt')

print(train_labels[0:5])
speakers_ids = np.unique(train_labels[:, 1])
print('\nnumber of speakers: ', speakers_ids.shape[0])

print(speakers_ids[0:5])
random_idxs = np.random.permutation(speakers_ids.shape[0])

speakers_ids = speakers_ids[random_idxs]
print(speakers_ids[0:5])

x_valid = KFold(n_splits=5)
#współczynnikiważenia
for train_idxs, test_idxs in x_valid.split(speakers_ids):


