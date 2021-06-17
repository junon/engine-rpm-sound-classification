from python_speech_features import mfcc, logfbank
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
import os


def plot_signals(signals):
    fig, axes = plt.subplots(nrows=5, ncols=1, sharex=False,sharey=False, figsize=(20,20))
    fig.suptitle('Time Series', size=16)
    for y in range(5):
        plt.subplot(5, 1, y+1)
        plt.plot(signals[y])

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=5, ncols=1, sharex=False,sharey=False, figsize=(20,20))
    fig.suptitle('Fourier Transforms', size=16)
    for y in range(5):
        plt.subplot(5, 1, y+1)
        data = fft[y]
        Y, freq = data[0], data[1]
        plt.plot(freq, Y)

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=5, ncols=1, sharex=False,sharey=False, figsize=(20,20))
    fig.suptitle('Filter Bank Coefficients', size=16)
    for y in range(5):
        plt.subplot(5, 1, y+1)
        plt.imshow(fbank[y], cmap='hot', interpolation='nearest')

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=5, ncols=1, sharex=False,sharey=False, figsize=(20,20))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    for y in range(5):
        plt.subplot(5, 1, y+1)
        plt.imshow(mfccs[y], cmap='hot', interpolation='nearest')

# def envelope(y, rate, threshold):
    # mask = []
    # y = pd.Series(y).apply(np.abs)
    # y_mean = y.rolling(window=int(rate/10), min_periods=1,center=True).mean()
    # for mean in y_mean:
    #     if mean > threshold:
    #         mask.append(True)
    #     else:
    #         mask.append(False)
    # return mask
    
def calc_fft(y,rate):
    n = len(y)
    freq = np.fft.rfftfreq(n,d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return(Y,freq)

signals = {}
fft = {}
fbank = {}
mfccs = {}

df = pd.read_csv('engine2/engine3.csv')

for c in range(len(df.fname)):
   signal,rate = librosa.load('engine2/' + df.fname[c],sr=16000)
   # mask = envelope(signal,rate,0.0005)
   # signal = signal[mask]
   signals[c]=signal
   fft[c]=calc_fft(signal,rate)
   bank = logfbank(signal[:rate],rate,nfilt=26,nfft=1103).T
   fbank[c] = bank
   print("fbank shape: {}".format(bank.shape))
   mel = mfcc(signal[:rate],rate,numcep=13, nfilt=26, nfft=1103).T
   mfccs[c] = mel
   print("mel shape: {}".format(mel.shape))

plot_signals(signals)
plt.show()
plot_fft(fft)
plt.show()
plot_fbank(fbank)
plt.show()
plot_mfccs(mfccs)
plt.show()

if len(os.listdir('engine2/clean'))==0:
    for f in tqdm(range(len(df.fname))):
        signal,rate = librosa.load('engine2/' + df.fname[f],sr=16000)
        # mask = envelope(signal,rate,0.0005)
        wavfile.write(filename='engine2/clean/'+df.fname[f],rate=rate,data=signal)
