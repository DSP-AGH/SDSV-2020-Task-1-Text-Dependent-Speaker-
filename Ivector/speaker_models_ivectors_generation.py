from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import ivector_engine.wave2ivec as w2i
from ivector_engine import utils
import numpy as np

i_vector = w2i.IVector()

from tqdm import tqdm

model_enrollment = utils.text_file_to_np_array("./docs/model_enrollment.txt")

# for row in tqdm(range(model_enrollment.shape[0])):
#     enroll_data = np.array([])
#     for it in range(2, 5):
#         fs, tmp = wavfile.read('./wav/enrollment/'+str(model_enrollment[row, it])+'.wav')
#         tmp = tmp/(2**15)
#         tmp = librosa.core.resample(tmp, fs, 8000)
#         enroll_data = np.concatenate((enroll_data, tmp), axis=0)
#
#     w, n_data = i_vector.process_wav(wav_file=(enroll_data, 8000), wav_type='data')
#     i_vector.save_ivector_to_file('./evaluation/model_enrollment/'+str(model_enrollment[row, 0])+'.ivec',
#                                   w, n_data)

