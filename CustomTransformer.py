import numpy as np
import warnings
from python_speech_features import mfcc, delta
from sklearn import preprocessing
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings('ignore')
from sklearn.base import BaseEstimator, TransformerMixin


def __init__(self, winlen=0.020, preemph=0.95, numcep=20, nfft=1024, ceplifter=15, highfreq=6000, nfilt=55,
             appendEnergy=False):
    self.winlen = winlen
    self.preemph = preemph
    self.numcep = numcep
    self.nfft = nfft
    self.ceplifter = ceplifter
    self.highfreq = highfreq
    self.nfilt = nfilt
    self.appendEnergy = appendEnergy

    def transform(self, x):
        """ A reference implementation of a transform function.
                Parameters
                ----------
                x : {array-like, sparse-matrix}, shape (n_samples, n_features)
                    The input samples.
                Returns
                -------
                X_transformed : array, shape (n_samples, n_features)
                    The array containing the element-wise square roots of the values
                    in ``X``.
                """

        # Check is fit has been called
        check_is_fitted(self, 'n_features_')

        # Check that the input is of the same shape as the one passed
        # during fit.
        if x.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        return self.signal_to_mfcc(x)

    def fit(self, x, y=None):
        """A reference implementation of a fitting function for a transformer.
                Parameters
                ----------
                x : {array-like, sparse matrix}, shape (n_samples, n_features)
                    The training input samples.
                y : None
                    There is no need of a target in a transformer, yet the pipeline API
                    requires this parameter.
                Returns
                -------
                self : object
                    Returns self.
                """

        self.n_features_ = x.shape[1]
    return self