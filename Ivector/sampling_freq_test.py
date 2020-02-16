from scipy.io import wavfile
import numpy as np
from ivector_engine import utils
from tqdm import tqdm

model_enrollment = utils.text_file_to_np_array('./docs/model_enrollment.txt')
train_labels = utils.text_file_to_np_array('./docs/train_labels.txt')
trials = utils.text_file_to_np_array('./docs/trials.txt')

model_enrollment_fs_list = np.array([])
train_labels_fs_list = np.array([])
trials_fs_list = np.array([])

for row in tqdm(range(model_enrollment.shape[0])):
    for it in range(2, 5):
        fs, _ = wavfile.read('./wav/enrollment/'+str(model_enrollment[row, it])+'.wav')
        model_enrollment_fs_list = np.append(model_enrollment_fs_list, fs)

model_enrollment_fs_mean = np.mean(model_enrollment_fs_list)
model_enrollment_fs_std_dev = np.std(model_enrollment_fs_list)
print('model_enrollment mean = ', model_enrollment_fs_mean)
print('model_enrollment std dev = ', model_enrollment_fs_std_dev)
del model_enrollment_fs_mean, model_enrollment_fs_std_dev, model_enrollment_fs_list

# for row in tqdm(range(train_labels.shape[0])):
#     fs, _ = wavfile.read('./wav/train/'+str(train_labels[row, 0])+'.wav')
#     train_labels_fs_list = np.append(train_labels_fs_list, fs)
#
# train_labels_fs_mean = np.mean(train_labels_fs_list)
# train_labels_fs_std_dev = np.std(train_labels_fs_list)
# print('train_labels mean = ', train_labels_fs_mean)
# print('train_labels std dev = ', train_labels_fs_std_dev)
# del train_labels_fs_mean, train_labels_fs_std_dev, train_labels_fs_list

# for row in tqdm(range(trials.shape[0])):
#     fs, _ = wavfile.read('./wav/evaluation/'+str(trials[row, 1])+'.wav')
#     trials_fs_list = np.append(trials_fs_list, fs)
#
# trials_fs_mean = np.mean(trials_fs_list)
# trials_fs_std_dev = np.std(trials_fs_list)
# print('trials mean = ', trials_fs_mean)
# print('trials std dev = ', trials_fs_std_dev)
# del trials_fs_mean, trials_fs_std_dev, trials_fs_list