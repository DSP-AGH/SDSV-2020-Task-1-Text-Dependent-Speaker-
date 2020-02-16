
import sys

import subprocess
import os
import errno

import numpy as np
import scipy.io.wavfile as spiowav

import ivector_engine.features as features
import ivector_engine.ivector as iv
import ivector_engine.gmm
import ivector_engine.energy_vad as evad
import ivector_engine.ivector_io as ivio


################################################################################
################################################################################

SOURCERATE    = 1250   
TARGETRATE    = 100000
LOFREQ        = 120
HIFREQ        = 3800

ZMEANSOURCE   = True
WINDOWSIZE    = 250000.0
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

################################################################################
################################################################################

class NoVadException(Exception):
    """ No VAD exception - raised when there is no VAD definition for a file
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def row(v):
  return v.reshape((1, v.size))


def mkdir_p(path):
    """ mkdir 
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def load_ubm(fname):
    """ This function will load the UBM from the file and will return the
        parameters in three separate variables
    """
    gmm = np.loadtxt(fname, dtype=np.float32)

    n_superdims = (gmm.shape[1] - 1) / 2

    weights = gmm[:,0]
    means   = gmm[:,1:(int(n_superdims)+1)]
    covs    = gmm[:,(int(n_superdims)+1):]

    return weights, means, covs


def load_vad_lab_as_bool_vec(lab_file):
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


def compute_vad(s, win_length=160, win_overlap=80):
    v = evad.compute_vad(s,win_length=win_length, win_overlap=win_overlap, n_realignment=10)

    n_frames = sum(v)
    n_regions = n_frames

    return v, n_regions, n_frames


def split_seq(seq,size):
    """ Split up seq in pieces of size """
    return [seq[i:i+size] for i in range(0, len(seq), size)]


def normalize_stats(n, f, ubm_means, ubm_norm):
    """ Center the first-order UBM stats around UBM means and normalize 
        by the UBM covariance 
    """
    n_gauss     = n.shape[0]
    n_superdim  = f.shape[0]
    n_fdim      = int(n_superdim / n_gauss)

    f0 = f  - ubm_means*np.kron(np.ones((n_fdim,1),dtype=n.dtype),n).transpose()
    f0 = f0 * ubm_norm

    return n, f0


################################################################################
################################################################################
fbank_mx      = features.mel_fbank_mx(winlen_nfft = WINDOWSIZE/SOURCERATE,
                                      fs          = fs,
                                      NUMCHANS    = NUMCHANS,
                                      LOFREQ      = LOFREQ,
                                      HIFREQ      = HIFREQ)

scp_list = "testme.scp"
vad_dir  = "auto"
wav_dir  = "./wav1"
ubm_file = "GMM.txt.gz"
v_file   = "v600_iter10.txt.gz"
out_dir  = "./out"

print('Loading UBM from' + str(ubm_file))
ubm_weights, ubm_means, ubm_covs = load_ubm(ubm_file)
GMM = ivector_engine.gmm.gmm_eval_prep(ubm_weights, ubm_means, ubm_covs)

numG=ubm_means.shape[0]
dimF=ubm_means.shape[1]

#normalization of statistics - precomputing matrices
if ubm_covs.shape[1] == dimF:
  ubm_norm = 1/np.sqrt(ubm_covs);

print('Loading T matrix from '+str(v_file)+'...')
v    = np.loadtxt(v_file, dtype=np.float32)

print('Computing MVVT ...')
MVVT = iv.compute_VtV(v, numG)

print('Loading list of files to process from ' + str(scp_list))
seg_list = np.atleast_1d(np.loadtxt(scp_list, dtype=object))

# extract all sub-dir names
for dir in set(map(os.path.dirname, seg_list)):
   mkdir_p(out_dir+'/'+dir)

# go over the scp and process the audio files
for ii, fn in enumerate(seg_list, 1):
    try:
        print('Processing ', str(ii), '/', str(len(seg_list)), str(fn))
        np.random.seed(777)

        wav_file      = wav_dir+'/'+fn+ '.wav'
        raw_file      = wav_dir+'/'+fn+ '.raw'
        lab_file      = vad_dir+'/'+fn+'.lab.gz'
        ivec_out_file = out_dir+'/'+fn+ '.ivec'

        

        if os.path.isfile(wav_file):
            rate, sig = spiowav.read(wav_file)
            print('  Reading wave file from ' + str(wav_file) + str(rate) + str(sig))

            if rate != 8000:
                raise Exception('The input file ' + wav_file + ' is expected to be in 8000 Hz sampling rate, but ' +repr(rate) + ' Hz detected')

        else:
            print('  Reading raw 8000Hz, 16bit-s, 1c,  file from ' + str(raw_file), 
            str(sig = np.fromfile(raw_file, dtype='int16')))

        print('[t=' + str(repr(len(sig) / fs)) + ' seconds, fs=' + str(repr(fs)) + 'Hz, n=' + str(repr(len(sig))) + ' samples]')

        if ADDDITHER > 0.0:
            print('  Adding dither')
            sig = features.add_dither(sig, ADDDITHER)

        fea = features.mfcc_htk(sig, 
                                window      = int(WINDOWSIZE/SOURCERATE),
                                noverlap    = int((WINDOWSIZE-TARGETRATE)/SOURCERATE),
                                fbank_mx    = fbank_mx,
                                _0          = 'first',
                                NUMCEPS     = NUMCEPS,
                                RAWENERGY   = RAWENERGY,
                                PREEMCOEF   = PREEMCOEF,
                                CEPLIFTER   = CEPLIFTER,
                                ZMEANSOURCE = ZMEANSOURCE,
                                ENORMALISE  = ENORMALISE,
                                ESCALE      = 0.1,
                                SILFLOOR    = 50.0,
                                USEHAMMING=True)
        print('  Extracting features' + str(fea))

        print('[n=' + str(repr(len(fea))) + ' frames]')

        print('  Adding derivatives')
        # [add_deriv] step 
        fea = features.add_deriv(fea,(deltawindow,accwindow))

        print('  Reshaping to SFeaCat convention')
        # [reshape] step 
        fea = fea.reshape(fea.shape[0], 3, -1).transpose((0,2,1)).reshape(fea.shape[0],-1) #re-order coeffs like SFeaCut

        if vad_dir == "auto":
            print('  Computing VAD ')
            vad,n_regions,n_frames = compute_vad(sig, win_length=WINDOWSIZE/SOURCERATE, win_overlap=(WINDOWSIZE-TARGETRATE)/SOURCERATE) [:len(fea)]
        else:
            print('  Loading VAD definition from ' + str(lab_file))
            vad,n_regions,n_frames = load_vad_lab_as_bool_vec(lab_file) [:len(fea)]
        
        print('  Applying VAD [#frames=' + repr(n_frames) + ', #regions=' + repr(n_regions) + ']')
        fea = fea[vad,...]

        if len(fea) < 3:
            raise NoVadException('Too few frames left: ' + str(len(fea)))

        print('  Applying floating CMVN')
        fea = features.cmvn_floating(fea, cmvn_lc, cmvn_rc, unbiased=True)

        n_data, d_data = fea.shape

        l        = 0;
        lc       = 0
        n        = np.zeros((numG),      dtype=np.float32)
        f        = np.zeros((numG,dimF), dtype=np.float32)

        seq_data = split_seq(range(n_data), 1000)
        print('  Computing stats ...' + str(seq_data))
        # Note that we compute the stats in in sub-chunks due to memory optimization
        #
        
        for i in range(len(seq_data)):
            dd = fea[seq_data[i],:]
            l1, n1, f1 = ivector_engine.gmm.gmm_eval(dd, GMM, return_accums=1)
            l  = l + l1.sum()
            lc = lc + l1.shape[0]
            n  = n + n1;
            f  = f + f1;

        print('[avg llh=' + repr(l/lc) + ', #frames=' + repr(n_data) + ']')

        n,f = normalize_stats(n, f, np.array(ubm_means).astype("int64"), np.array(ubm_norm).astype("int64"))

        f   = row(f.astype(v.dtype))
        n   = row(n.astype(v.dtype))

        print('  Computing i-vector')
        w   = iv.estimate_i(n, f, v, MVVT).T
        
        #write it to the disk
        print('  Saving ivec to:', str(ivec_out_file))
        # np.savetxt(ivec_out_file, w.ravel(), newline=' ', fmt='%f')
        print(type(w))
        ivio.write_binary_ivector(ivec_out_file, w.ravel(), int(n_data/100.0))

    except NoVadException as e:
        print(e)
        print("Warning: No features generated for segment: "  +str(fn))

    except:
        raise


