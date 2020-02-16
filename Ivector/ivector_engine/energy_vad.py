import ivector_engine.gmm
import ivector_engine.features
import numpy as np


def compute_vad(s, win_length=160, win_overlap=80, n_realignment=5, threshold=0.3):
    # power signal for energy computation
    s = s**2

    # frame signal with overlap
    F = ivector_engine.features.framing(s, win_length, win_length - win_overlap)

    # sum frames to get energy
    E = F.sum(axis=1)

    # E = np.sqrt(E)
    # E = np.log(E)

    # normalize the energy
    E -= E.mean()
    E /= E.std()


    # initialization
    mm = np.array((-1.00, 0.00, 1.00))[:, np.newaxis]
    ee = np.array(( 1.00, 1.00, 1.00))[:, np.newaxis]
    ww = np.array(( 0.33, 0.33, 0.33))

    GMM = ivector_engine.gmm.gmm_eval_prep(ww, mm, ee)

    E = E[:,np.newaxis]

    for i in range(n_realignment):
        # collect GMM statistics
        llh, N, F, S = ivector_engine.gmm.gmm_eval(E, GMM, return_accums=2)

        # update model
        ww, mm, ee   = ivector_engine.gmm.gmm_update(N, F, S)

        # wrap model
        GMM = ivector_engine.gmm.gmm_eval_prep(ww, mm, ee)

    # evaluate the gmm llhs
    llhs = ivector_engine.gmm.gmm_llhs(E, GMM)

    llh  = ivector_engine.gmm.logsumexp(llhs, axis=1)[:, np.newaxis]

    llhs = np.exp(llhs - llh)

    out  = np.zeros(llhs.shape[0], dtype=np.bool)
    out[llhs[:,0] < threshold] = True

    return out
    

