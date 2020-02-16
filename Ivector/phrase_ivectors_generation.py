from scipy.io import wavfile
import ivector_engine.wave2ivec as w2i
from ivector_engine import utils
import numpy as np
from tqdm import tqdm


i_vector = w2i.IVector()

train_labels = utils.text_file_to_np_array("./docs/train_labels.txt")

phrases_ivectors = []
phrases_ids = []

for row in tqdm(range(int(train_labels.shape[0] // 20))):
    fs, data = wavfile.read('./wav/train/'+str(train_labels[row, 0])+'.wav')
    data = data/(2**15)
    w, n_data = i_vector.process_wav(wav_file=(data, fs), wav_type='data')
    phrases_ivectors.append(w)
    phrases_ids.append(int(train_labels[row, 2]))
    # i_vector.save_ivector_to_file(
    #     './evaluation/phrases_ivectors/ivector_'+str(train_labels[row, 0][4:]) +
    #     '_phrase_'+str(train_labels[row, 2])+'.ivec',
    #     w, n_data)

phrases_ids = np.array(phrases_ids)
phrases_ivectors = np.array(phrases_ivectors)
np.savez_compressed('phrases_ivectors.npz', phrases_ivectors = phrases_ivectors, phrases_ids = phrases_ids)