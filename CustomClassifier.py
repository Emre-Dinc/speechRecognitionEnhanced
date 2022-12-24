import glob
import os

import numpy as np
import warnings

from kaldiio.wavio import read_wav
from python_speech_features import mfcc, delta
from scipy.io import wavfile
from sklearn import preprocessing
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings('ignore')
from sklearn.base import BaseEstimator, TransformerMixin


def load_data(folder):
    x, y = [], []

    for voice_sample in list(glob.glob(rf'./{folder}/id*/id*')):
        voice_sample_file_name = os.path.basename(voice_sample)
        voice_class, _ = voice_sample_file_name.split("_")

        features = read_wav(voice_sample)

        x.append(features)
        y.append(voice_class)

    return np.array(x, dtype=tuple), np.array(y)


def read_wav(fname):
    fs, signal = wavfile.read(fname)
    if len(signal.shape) != 1:
        signal = signal[:, 0]
    return fs, signal
