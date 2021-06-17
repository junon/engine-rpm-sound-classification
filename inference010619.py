from python_speech_features import mfcc
#from keras.models import load_model
import pandas as pd
import numpy as np
import librosa
import pickle
import os
import tensorflow as tf


class Config:
    def __init__ (self,mode='conv',nfilt=26,nfeat=13,nfft=512,rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)
        self.model_path = os.path.join('engine2/models',mode + '.model')
        self.p_path = os.path.join('engine2/pickles', mode + '.p')

# def envelope(y, rate, threshold):
#     mask = []
#     y = pd.Series(y).apply(np.abs)
#     y_mean = y.rolling(window=int(rate/10), min_periods=1,center=True).mean()
#     for mean in y_mean:
#         if mean > threshold:
#             mask.append(True)
#         else:
#             mask.append(False)
#     return mask

p_path = os.path.join('engine2/pickles','conv.p')
with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
model = tf.keras.models.load_model("engine2/models/conv.model")

categories=['1000rpm','2000rpm','3000rpm']

signal,rate = librosa.load("/home/iotg-mmml/ConvNet/engine2/testFile.wav",sr=16000)
# mask = envelope(signal,rate,0.0005)
wav = signal #[mask]

for i in range(0,wav.shape[0]-config.step,config.step):
    sample = wav[i:i+config.step]
    x = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
    x = (x-config.min)/(config.max-config.min)
    if config.mode =='conv':
        x = x.reshape(1, x.shape[0], x.shape[1], 1)
    score = model.predict(x)
    print("Category: {}, Confidence: {:.4f}".format(categories[np.argmax(score[0])],score[0][np.argmax(score[0])]))
