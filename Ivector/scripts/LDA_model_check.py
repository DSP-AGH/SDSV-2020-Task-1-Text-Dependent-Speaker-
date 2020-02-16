import pickle
import numpy as np

with open('../phrases_ivectors/LDA_model.pickle', 'rb') as file:
    clf = pickle.load(file)


test_ivector = np.expand_dims(np.random.rand(600), axis=0)
test_trans = clf.transform(test_ivector)
print(test_trans)