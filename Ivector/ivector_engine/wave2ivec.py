import sys

import subprocess
import os
import errno

import numpy as np

import ivector_engine.features
import ivector_engine.ivector as iv
import ivector_engine.gmm as gmm
import ivector_engine.energy_vad as evad
import ivector_engine.ivector_io as ivio

import librosa
import zipfile
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# CONSTANTS
SOURCERATE    = 1250
TARGETRATE    = 100000
LOFREQ        = 120
HIFREQ        = 3800

ZMEANSOURCE   = True
WINDOWSIZE    = 250000
USEHAMMING    = True
PREEMCOEF     = 0.97
NUMCHANS      = 24
CEPLIFTER     = 22
NUMCEPS       = 19
ADDDITHER     = 1.0
RAWENERGY     = True
ENORMALISE    = True

deltawindow   = accwindow = 2

cmvn_lc       = 150
cmvn_rc       = 150

fs            = 1e7/SOURCERATE

fbank_mx      = ivector_engine.features.mel_fbank_mx(winlen_nfft =WINDOWSIZE / SOURCERATE,
                                                     fs          = fs,
                                                     NUMCHANS    = NUMCHANS,
                                                     LOFREQ      = LOFREQ,
                                                     HIFREQ      = HIFREQ)

class NoVadException(Exception):
    """ No VAD exception - raised when there is no VAD definition for a file
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class IVector():
    
    def __init__(self):
        print("INICJALIZACJA!")

        # MODELS
        ubm_file = "./models/GMM.txt.gz"
        v_file = "./models/v600_iter10.txt.gz"

        print('Loading UBM from', ubm_file)
        ubm_weights, ubm_means, ubm_covs = self.load_ubm(ubm_file)
        GMM = gmm.gmm_eval_prep(ubm_weights, ubm_means, ubm_covs)

        numG = ubm_means.shape[0]
        dimF = ubm_means.shape[1]

        # normalization of statistics - precomputing matrices
        if ubm_covs.shape[1] == dimF:
            ubm_norm = 1 / np.sqrt(ubm_covs);

        print('Loading T matrix from ' + str(v_file) + '...')
        #v = np.loadtxt(v_file, dtype=np.float32)
        v = pd.read_csv(v_file, sep=" ", header=None, dtype=np.float32)
        v = v.to_numpy()

        print('Computing MVVT ...')
        MVVT = iv.compute_VtV(v, numG)

        self.v = v
        self.MVVT = MVVT
        self.GMM = GMM
        self.numG = numG
        self.dimF = dimF
        self.ubm_norm = ubm_norm
        self.ubm_means = ubm_means

    def row(self, v):
      return v.reshape((1, v.size))

    def mkdir_p(self, path):
        """ mkdir
        """
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise


    def load_ubm(self, fname):
        """ This function will load the UBM from the file and will return the
            parameters in three separate variables
        """
        gmm = np.loadtxt(fname, dtype=np.float32)

        n_superdims = (gmm.shape[1] - 1) / 2

        weights = gmm[:,0]
        means   = gmm[:,1:(int(n_superdims)+1)]
        covs    = gmm[:,(int(n_superdims)+1):]

        return weights, means, covs


    def load_vad_lab_as_bool_vec(self, lab_file):
        lab_cont = np.atleast_2d(np.loadtxt(lab_file, dtype=object))

        if lab_cont.shape[1] == 0:
            return np.empty(0), 0, 0

        # else:
        #     lab_cont = lab_cont.reshape((-1,lab_cont.shape[0]))

        if lab_cont.shape[1] == 3:
            lab_cont = lab_cont[lab_cont[:,2]=='sp',:][:,[0,1]]

        n_regions = lab_cont.shape[0]

        vad     = np.round(np.atleast_2d(lab_cont).astype(np.float).T * 100).astype(np.int)
        vad[1] += 1 #Paja's bug!!!

        if not vad.size:
            return np.empty(0, dtype=bool)

        npc1 = np.c_[np.zeros_like(vad[0], dtype=bool), np.ones_like(vad[0], dtype=bool)]
        npc2 = np.c_[vad[0] - np.r_[0, vad[1,:-1]], vad[1]-vad[0]]

        out  = np.repeat(npc1, npc2.flat)

        n_frames = sum(out)

        return out, n_regions, n_frames


    def compute_vad(self, s, win_length=160, win_overlap=80):
        v = evad.compute_vad(s,win_length=win_length, win_overlap=win_overlap, n_realignment=10)

        n_frames = sum(v)
        n_regions = n_frames

        return v, n_regions, n_frames


    def split_seq(self, seq,size):
        """ Split up seq in pieces of size """
        return [seq[i:i+size] for i in range(0, len(seq), size)]


    def normalize_stats(self, n, f, ubm_means, ubm_norm):
        """ Center the first-order UBM stats around UBM means and normalize
            by the UBM covariance
        """
        n_gauss     = n.shape[0]
        n_superdim  = f.shape[0]
        n_fdim      = int(n_superdim / n_gauss)

        f0 = f  - ubm_means*np.kron(np.ones((n_fdim,1),dtype=n.dtype),n).transpose()
        f0 = f0 * ubm_norm

        return n, f0

    def load_gzvectors_into_ndarray(self, lst, prefix='', suffix='', dtype=np.float64):
        """ Loads the scp list into ndarray
        """
        n_data = lst.shape[0]
        v_dim = None

        for ii, segname in enumerate(lst):
            print('Loading [{}/{}] {}'.format(ii, n_data, segname))

            tmp_vec = np.loadtxt(prefix + segname + suffix, dtype=dtype)

            if v_dim == None:
                v_dim = len(tmp_vec)
                out = np.zeros((n_data, v_dim), dtype=dtype)
            elif v_dim != len(tmp_vec):
                raise ValueError(str.format("Vector {} is of wrong size ({} instead of {})",
                                            segname, len(tmp_vec), v_dim))

            out[ii, :] = tmp_vec

        return out

    ################################################################################
    ################################################################################
    def load_vectors_into_ndarray(self, lst, prefix='', suffix='', dtype=np.float64):
        """ Loads the scp list into ndarray
        """
        n_data = lst.shape[0]
        v_dim = None

        for ii, segname in enumerate(lst):
            print('Loading [{}/{}] {}'.format(ii, n_data, segname))

            tmp_vec, n_frames, tags = ivio.read_binary_ivector(prefix + segname + suffix)

            if v_dim == None:
                v_dim = len(tmp_vec)
                out = np.zeros((n_data, v_dim), dtype=dtype)
            elif v_dim != len(tmp_vec):
                raise ValueError(str.format("Vector {} is of wrong size ({} instead of {})",
                                            segname, len(tmp_vec), v_dim))

            out[ii, :] = tmp_vec

        return out

    ################################################################################
    ################################################################################
    def warp2us(self, ivecs, lda, lda_mu):
        """ i-vector pre-processing
            This function applies a global LDA, mean subtraction, and length
            normalization.
        """
        ivecs = ivecs.dot(lda) - lda_mu
        ivecs /= np.sqrt((ivecs ** 2).sum(axis=1)[:, np.newaxis])
        return ivecs

    ################################################################################
    ################################################################################
    def bilinear_plda(self, Lambda, Gamma, c, k, Fe, Ft):
        """ Performs a full PLDA scoring
        """
        out = np.empty((Fe.shape[0], Ft.shape[0]), dtype=Lambda.dtype)

        np.dot(Fe.dot(Lambda), Ft.T, out=out)
        out += (np.sum(Fe.dot(Gamma) * Fe, 1) + Fe.dot(c))[:, np.newaxis]
        out += (np.sum(Ft.dot(Gamma) * Ft, 1) + Ft.dot(c))[np.newaxis, :] + k

        return out

    def cosine_distance_score(self, enroll_ivec, test_ivec):
        """Performs full cds scoring
        """
        out = []
        for enroll_it in range(enroll_ivec.shape[0]):
            row_list = []
            for test_it in range(test_ivec.shape[0]):
                unit_cds = cosine_similarity(enroll_ivec[enroll_it, :], test_ivec[test_it, :])
                row_list.append(unit_cds)
            out.append(row_list)

        return out



    def wav_conversion(self, y, fs):
        y = librosa.resample(y, fs, 8000)
        fs = 8000
        return y, fs

    def process_wav(self, wav_file, wav_type='data', mode="ivector", vad_dir="auto"):
        if mode not in ["ivector", "statistics", "mfcc"]:
            return False

        else:
            # all constans are initialized in __init__() method
            # READ WAVE AND COMPUTE IVECTOR
            #modification
            # if wav_type = data, argument should be a tuple (signal, fs)
            # i wav_type = path, then data is imported from path
            if wav_type == 'path':
                sig, rate = librosa.load(wav_file)
            elif wav_type == 'data':
                sig, rate = wav_file
            else:
                return False
            
            
            # wav conversion
            sig, rate = self.wav_conversion(sig, rate)

            if rate != 8000:
                raise Exception('The input file ' + wav_file + ' is expected to be in 8000 Hz sampling rate, but ' + repr(
                    rate) + ' Hz detected')

            # info about singnal printed
            #print('[t=' + repr(len(sig) / fs) + ' seconds, fs=' + repr(rate) + 'Hz, n=' + repr(len(sig)) + ' samples]')

            if ADDDITHER > 0.0:
                #print('  Adding dither')
                sig = ivector_engine.features.add_dither(sig, ADDDITHER)

           
            fea = ivector_engine.features.mfcc_htk(sig,
                                                   window=WINDOWSIZE / SOURCERATE,
                                                   noverlap=(WINDOWSIZE - TARGETRATE) / SOURCERATE,
                                                   fbank_mx=fbank_mx,
                                                   _0='first',
                                                   NUMCEPS=NUMCEPS,
                                                   RAWENERGY=RAWENERGY,
                                                   PREEMCOEF=PREEMCOEF,
                                                   CEPLIFTER=CEPLIFTER,
                                                   ZMEANSOURCE=ZMEANSOURCE,
                                                   ENORMALISE=ENORMALISE,
                                                   ESCALE=0.1,
                                                   SILFLOOR=50.0,
                                                   USEHAMMING=True)
            #print('  Extracting features' + str(fea))
            #print('[n=' + repr(len(fea)) + ' frames]')

            #print('  Adding derivatives')
            # [add_deriv] step
            fea = ivector_engine.features.add_deriv(fea, (deltawindow, accwindow))

            #print('  Reshaping to SFeaCat convention')
            # [reshape] step
            fea = fea.reshape(fea.shape[0], 3, -1).transpose((0, 2, 1)).reshape(fea.shape[0],
                                                                                -1)  # re-order coeffs like SFeaCut
            if vad_dir == "auto":
                #print('  Computing VAD ')
                vad, n_regions, n_frames = self.compute_vad(sig, win_length=WINDOWSIZE / SOURCERATE,
                                                       win_overlap=(WINDOWSIZE - TARGETRATE) / SOURCERATE)[:len(fea)]

                #print('  Applying VAD [#frames=' + repr(n_frames) + ', #regions=' + repr(n_regions) + ']')
                fea = fea[0:len(vad), ...]
                fea = fea[vad, ...]

                if len(fea) < 3:
                    raise NoVadException('Too few frames left: ' + str(len(fea)))

                #print('  Applying floating CMVN')
                fea = ivector_engine.features.cmvn_floating(fea, cmvn_lc, cmvn_rc, unbiased=True)

                if mode == "mfcc":
                    return fea

                n_data, d_data = fea.shape

                l = 0;
                lc = 0
                n = np.zeros((self.numG), dtype=np.float32)
                f = np.zeros((self.numG, self.dimF), dtype=np.float32)

               # print('  Computing stats ...')
                # Note that we compute the stats in in sub-chunks due to memory optimization
                #
                seq_data = self.split_seq(range(n_data), 1000)
                #print('  Computing stats ...' + str(seq_data))
                for i in range(len(seq_data)):
                    dd = fea[seq_data[i], :]
                    l1, n1, f1 = gmm.gmm_eval(dd, self.GMM, return_accums=1)
                    l = l + l1.sum()
                    lc = lc + l1.shape[0]
                    n = n + n1;
                    f = f + f1;

                #print('[avg llh=' + repr(l / lc) + ', #frames=' + repr(n_data) + ']')

                n, f = self.normalize_stats(n, f, self.ubm_means, self.ubm_norm)

                f = self.row(f.astype(self.v.dtype))
                n = self.row(n.astype(self.v.dtype))

                if mode == "statistics":
                    return f, n

                #print('  Computing i-vector')
                w = iv.estimate_i(n, f, self.v, self.MVVT).T

                #print("IVECTOR" + str(w))

                if mode == "ivector":
                    return w, n_data

    def save_ivector_to_file(self, path, ivector, n_data):
        #print('  Saving ivec to:' + str(path))
        #np.savetxt(path, ivector.ravel(), newline=' ', fmt='%f')
        ivio.write_binary_ivector(path, ivector.ravel(), int(n_data/100.0))

    def mfcc_to_ivector(self, fea):
        n_data, d_data = fea.shape

        l = 0;
        lc = 0
        n = np.zeros((self.numG), dtype=np.float32)
        f = np.zeros((self.numG, self.dimF), dtype=np.float32)

        
        # Note that we compute the stats in in sub-chunks due to memory optimization
        #
        seq_data = self.split_seq(range(n_data), 1000)
        print('  Computing stats ...' + str(seq_data))
        for i in range(len(seq_data)):
            dd = fea[seq_data[i], :]
            l1, n1, f1 = gmm.gmm_eval(dd, self.GMM, return_accums=1)
            l = l + l1.sum()
            lc = lc + l1.shape[0]
            n = n + n1;
            f = f + f1;

        print('[avg llh=' + repr(l / lc) + ', #frames=' + repr(n_data) + ']')

        n, f = self.normalize_stats(n, f, self.ubm_means, self.ubm_norm)

        f = self.row(f.astype(self.v.dtype))
        n = self.row(n.astype(self.v.dtype))

        print('  Computing i-vector')
        w = iv.estimate_i(n, f, self.v, self.MVVT).T

        print("IVECTOR" + str(w))

        return w

    def get_matrices(self, mode="TV"):
        if mode not in ["TV", "LDA", "PLDA"]:
            return False
        else:
            if mode == "TV":
                v_file = "../models/v600_iter10.txt.gz"
                v = np.loadtxt(v_file, dtype=np.float32)
                return v
            elif mode == "LDA":
                v_file = "../models/backend/backend.LDA.txt.gz"
                v = np.loadtxt(v_file, dtype=np.float32)
                return v
            elif mode == "PLDA_c":
                v_file = "../models/backend/backend.PLDA.c.txt.gz"
                v = np.loadtxt(v_file, dtype=np.float32)
                return v
            elif mode == "PLDA_gamma":
                v_file = "../models/backend/backend.PLDA.Gamma.txt.gz"
                v = np.loadtxt(v_file, dtype=np.float32)
                return v
            elif mode == "PLDA_k":
                v_file = "../models/backend/backend.PLDA.k.txt.gz"
                v = np.loadtxt(v_file, dtype=np.float32)
                return v
            elif mode == "PLDA_lambda":
                v_file = "../models/backend/backend.PLDA.Lambda.txt.gz"
                v = np.loadtxt(v_file, dtype=np.float32)
                return v

    # we get scoring from 2 the same ivectors
    def get_score_from_ivectors(self, path_enroll, path_test, plda_model_dir="../models/backend"):
        # reading models from directories
        plda_model_dir = "./models/backend"

        lda_file = plda_model_dir + '/backend.LDA.txt.gz'
        mu_file = plda_model_dir + '/backend.mu_train.txt.gz'
        Gamma_file = plda_model_dir + '/backend.PLDA.Gamma.txt.gz'
        Lambda_file = plda_model_dir + '/backend.PLDA.Lambda.txt.gz'
        c_file = plda_model_dir + '/backend.PLDA.c.txt.gz'
        k_file = plda_model_dir + '/backend.PLDA.k.txt.gz'

        lda = np.loadtxt(lda_file, dtype=np.float32)
        mu = np.loadtxt(mu_file, dtype=np.float32)
        Gamma = np.loadtxt(Gamma_file, dtype=np.float32)
        Lambda = np.loadtxt(Lambda_file, dtype=np.float32)
        c = np.loadtxt(c_file, dtype=np.float32)
        k = np.loadtxt(k_file, dtype=np.float32)

        files_enroll = []
        files_test = []
        # enrolled ivec
        #print(os.listdir(path))

        for file in os.listdir(path_enroll):
            if file.endswith(".ivec"):
                filenames = os.path.join(path_enroll, file)
                files_enroll.append(filenames)

        for file in os.listdir(path_test):
            if file.endswith(".ivec"):
                filenames = os.path.join(path_test, file)
                files_test.append(filenames)

        enroll_ivec = self.load_vectors_into_ndarray(np.asarray(files_enroll), prefix="", suffix="",
                                                        dtype=np.float32)

        test_ivec = self.load_vectors_into_ndarray(np.asarray(files_test), prefix="", suffix="",
                                                     dtype=np.float32)

        print('Transforming and normalizing i-vectors')
        enroll_ivec = self.warp2us(enroll_ivec, lda, mu)
        test_ivec = self.warp2us(test_ivec, lda, mu)


        print('Computing PLDA score')
        plda_table = self.bilinear_plda(Lambda, Gamma, c, k, enroll_ivec, test_ivec)
        print(plda_table)

        print('Computing CDS score')
        cds_table = cosine_similarity(enroll_ivec, test_ivec)



        return cds_table, plda_table

    # similar function but input files are in zip format
    def get_score_from_ivectors_zip(self, zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zip2:
            zip2.extractall("../zip")
        path = "../zip"
        return self.get_score_from_ivectors(path)

# if __name__ == '__main__':
#     o = IVector()
#     o.get_score_from_ivectors("../test/out/fisher-english-p1")



