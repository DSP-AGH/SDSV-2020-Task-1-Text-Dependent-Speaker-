import numpy as np
import matplotlib.pyplot as plt
from ivector_engine import utils
from sklearn.model_selection import KFold
from ivector_engine import wave2ivec as w2i
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from scipy.io import wavfile
from tqdm import tqdm
import pickle


i_vector = w2i.IVector()
train_labels = utils.text_file_to_np_array('./docs/train_labels.txt')

print(train_labels[0:5])
speakers_ids = np.unique(train_labels[:, 1])
phrases_ids = np.unique(train_labels[:, 2])
random_idxs = np.random.permutation(train_labels.shape[0])
shuffled_train_labels = train_labels[random_idxs]
train_set = shuffled_train_labels[:5000]
print('\nnumber of speakers: ', speakers_ids.shape[0])

print(speakers_ids[0:5])
random_idxs = np.random.permutation(speakers_ids.shape[0])

speakers_ids = speakers_ids[random_idxs]
print(speakers_ids[0:5])

x_valid = KFold(n_splits=5)
#współczynnikiważenia
print('Start of s-split crossvalidation')
split_count = 1
for train_idxs, test_idxs in x_valid.split(speakers_ids):
    #calculating training ivectors
    print('split ', str(split_count))
    print('calculating training ivectors...')
    train_ivectors = []
    train_phrase_list_for_ivectors = []
    for it in tqdm(range(train_set.shape[0])):
        if train_set[it, 1] in speakers_ids[train_idxs]:
            fs, data = wavfile.read('./wav/train/'+str(train_set[it, 0]+'.wav'))
            data = data / (2 ** 15)
            w, n_data = i_vector.process_wav(wav_file=(data, fs), wav_type='data')
            train_ivectors.append(w)
            train_phrase_list_for_ivectors.append(int(train_set[it, 2]))

    train_ivectors = np.squeeze(np.array(train_ivectors))
    train_phrase_list_for_ivectors = np.array(train_phrase_list_for_ivectors)
    print('train_ivectors shape:', train_ivectors.shape)
    print(train_phrase_list_for_ivectors.shape)
    #calculating LDA transformation for training ivectors, mean vector for every phrase
    print('calculating LDA transformation for training ivectors, mean vector for every phrase')
    clf = LinearDiscriminantAnalysis(priors=np.arange(1, 11), n_components=9, store_covariance=True)
    trans_train_ivectors = clf.fit_transform(train_ivectors, train_phrase_list_for_ivectors)

    print('saving LDA to pickle file...')
    with open('./crossvalidation_files/LDA/models/LDA_split_' + str(split_count) + '.pickle', 'wb') as file:
        pickle.dump(clf, file)


    trans_mean_phrase_ivectors = []
    for class_it in range(1, 11):
        trans_class_ivectors = []
        for row in range(trans_train_ivectors.shape[0]):
            if int(train_phrase_list_for_ivectors[row]) == class_it:
                trans_class_ivectors.append(trans_train_ivectors[row, :])

        trans_class_ivectors = np.squeeze(np.array([trans_class_ivectors]))
        trans_class_mean_ivector = np.mean(trans_class_ivectors, axis=0)
        trans_mean_phrase_ivectors.append(trans_class_mean_ivector)

    trans_mean_phrase_ivectors = np.array(trans_mean_phrase_ivectors)
    np.savez('./crossvalidation_files/LDA/mean_vectors/trans_mean_phrase_ivectors_split_'+str(split_count)+'.npz',
             phrase_ivectors=trans_mean_phrase_ivectors)
    print('trans_mean_phrase_ivectors shape', trans_mean_phrase_ivectors.shape)

    #training set validation
    train_corr_scores = []
    train_neg_scores = []
    for it in tqdm(range(trans_train_ivectors.shape[0])):
        for phrase_it in range(trans_mean_phrase_ivectors.shape[0]):
            phrase_vector = np.expand_dims(trans_mean_phrase_ivectors[phrase_it], axis=0)
            print('phrase_vector shape: ', phrase_vector.shape)
            if int(train_phrase_list_for_ivectors[it]) == phrase_it + 1:
                train_corr_scores.append(cosine_similarity(np.expand_dims(trans_train_ivectors[it], axis=0),
                                                           phrase_vector))
            else:
                train_neg_scores.append(cosine_similarity(np.expand_dims(trans_train_ivectors[it], axis=0),
                                                          phrase_vector))

    train_corr_scores = np.squeeze(np.array(train_corr_scores))
    train_neg_scores = np.squeeze(np.array(train_neg_scores))
    print('plotting histograms')
    plt.hist(train_neg_scores)
    plt.hist(train_corr_scores)
    title = 'train set - split ' + str(split_count)
    plt.title(title)
    plt.legend(['correct scores', 'negative_scores'])
    plt.show()

    # training scatterplots
    print("plotting scatterplots for training set")
    for phrase_it in phrases_ids:
        tmp_class_LDA = []
        for y_it in range(len(train_phrase_list_for_ivectors)):
            if train_phrase_list_for_ivectors[y_it] == phrase_it:
                tmp_class_LDA.append(trans_train_ivectors[y_it, :])
        tmp_class_LDA = np.array(tmp_class_LDA)
        print(tmp_class_LDA.shape)
        plt.scatter(tmp_class_LDA[:, 0], tmp_class_LDA[:, 1])
    plt.xlabel('LDA_1')
    plt.ylabel('LDA_2')
    plt.legend(phrases_ids)
    plt.title('training set scatter plot - split ' + str(split_count))
    plt.show()

    #calculating cosine similarity for test ivectors, appending correct and negative scores
    #test set validation
    print('calculating cosine similarity for test ivectors, appending correct and negative scores')
    trans_test_ivectors = []
    test_phrase_list_for_ivectors = []
    corr_scores = []
    neg_scores = []
    for it in tqdm(range(train_set.shape[0])):
        if train_set[it, 1] in speakers_ids[test_idxs]:
            fs, data = wavfile.read('./wav/train/' + str(train_set[it, 0] + '.wav'))
            data = data / (2 ** 15)
            w, n_data = i_vector.process_wav(wav_file=(data, fs), wav_type='data')
            trans_ivector = np.array(clf.transform(w))
            trans_test_ivectors.append(np.squeeze(trans_ivector).tolist())
            test_phrase_list_for_ivectors.append(int(train_set[it, 2]))
            target_phrase = train_set[it, 2]
            for phrase_it in range(trans_mean_phrase_ivectors.shape[0]):
                phrase_vector = np.expand_dims(trans_mean_phrase_ivectors[phrase_it], axis=0)
                if int(train_set[it, 2]) == phrase_it + 1:
                    corr_scores.append(cosine_similarity(trans_ivector, phrase_vector))
                else:
                    neg_scores.append(cosine_similarity(trans_ivector, phrase_vector))

    corr_scores = np.squeeze(np.array(corr_scores))
    neg_scores = np.squeeze(np.array(neg_scores))
    print('Single split calculations finished!')
    print('Saving histograms...')
    n_corr, bins_corr, patches_corr = plt.hist(corr_scores)
    np.savez_compressed('./crossvalidation_files/histogram_data/corr_histogram_split_'+str(split_count)+'.npz',
                        n_corr=n_corr, bins_corr=bins_corr)

    n_neg, bins_neg, patches_neg = plt.hist(neg_scores)
    np.savez_compressed('./crossvalidation_files/histogram_data/neg_histogram_split_'+str(split_count)+'.npz',
                        n_neg=n_neg, bins_neg=bins_neg)

    print('plotting histograms')
    plt.hist(neg_scores)
    plt.hist(corr_scores)
    title = 'split '+str(split_count)
    plt.title(title)
    plt.legend(['correct scores', 'negative_scores'])
    plt.show()

    trans_test_ivectors = np.array(trans_test_ivectors)
    print('trans_')
    # test scatterplots
    print("plotting scatterplots for test set")
    for phrase_it in phrases_ids:
        tmp_class_LDA = []
        for y_it in range(len(test_phrase_list_for_ivectors)):
            if test_phrase_list_for_ivectors[y_it] == phrase_it:
                tmp_class_LDA.append(trans_test_ivectors[y_it, :])
        tmp_class_LDA = np.array(tmp_class_LDA)
        print(tmp_class_LDA.shape)
        plt.scatter(tmp_class_LDA[:, 0], tmp_class_LDA[:, 1])
    plt.xlabel('LDA_1')
    plt.ylabel('LDA_2')
    plt.legend(phrases_ids)
    plt.title('test set scatter plot - split ' + str(split_count))
    plt.show()

    split_count+=1

