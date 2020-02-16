from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from tqdm import tqdm
import pickle


data = np.load('./phrases_ivectors.npz')
phrases_ivectors, phrases_ids = [data[f] for f in data]
phrases_ivectors = np.squeeze(phrases_ivectors)

clf = LinearDiscriminantAnalysis(priors=np.arange(1, 11), n_components=9, store_covariance=True)
trans_ivectors = clf.fit_transform(phrases_ivectors, phrases_ids)
print('trans_ivectors_shape: ', trans_ivectors.shape)

coef_matrix = clf.coef_
print('coef matrix shape:\n', coef_matrix.shape)

params = clf.get_params()
print(params)

filename = 'LDA_model.pickle'
with open(filename, 'wb') as file:
    pickle.dump(clf, file)

# trans_mean_ivectors = []
# for class_it in tqdm(range(1, 11)):
#     trans_class_ivectors = []
#     for row in range(trans_ivectors.shape[0]):
#         if phrases_ids[row] == class_it:
#             trans_class_ivectors.append(trans_ivectors[row, :])
#
#     trans_class_ivectors = np.squeeze(np.array([trans_class_ivectors]))
#     trans_class_mean_ivector = np.mean(trans_class_ivectors, axis=0)
#     trans_mean_ivectors.append(trans_class_mean_ivector)
#
# trans_mean_ivectors = np.array(trans_mean_ivectors)
#
# print(trans_mean_ivectors.shape)
#
# np.savez_compressed('./LDA_nine_dim_phrases_ivectors.npz', LDA_nine_dim_phrases_ivectors=trans_mean_ivectors)


