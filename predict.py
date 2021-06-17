from sklearn.metrics import accuracy_score
# from python_speech_features import mfcc
#from keras.models import load_model
# from scipy.io import wavfile
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels


def build_prediction(audio_dir):
    y_true = []
    y_pred = []
    fn_prob = {}

    print('extracting features from audio')
    for fn in tqdm(os.listdir(audio_dir)):
        # rate, wav = wavfile.read(os.path.join(audio_dir,fn))
        label = fn2class[fn]
        c = classes.index(label)
        y_prob = []
        for i in range(0,wav.shape[0]-config.step,config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
            x = (x-config.min)/(config.max-config.min)
            if config.mode =='conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
    return y_true, y_pred, fn_prob

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

df = pd.read_csv('engine2/engine3.csv')
classes = list(np.unique(df.category))
fn2class = dict(zip(df.filename, df.category)) #change here

p_path = 'engine2/pickles/conv.p'
with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
model = tf.keras.models.load_model('engine2/models/conv.model') #config.model_path
y_true, y_pred, fn_prob = build_prediction('engine2/clean')

plot_confusion_matrix(y_true, y_pred, classes=classes, title='Confusion matrix, without normalization')
plot_confusion_matrix(y_true, y_pred, classes=classes, normalize=True, title='Normalized confusion matrix')
plt.show()

acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)
y_probs = []
for i,row in df.iterrows():
    y_prob = fn_prob[row.filename] #change here
    y_probs.append(y_prob)

    for c,p in zip(classes, y_prob):
        df.at[i,c]=p
y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred

df.to_csv('engine2/predictionsEngine2.csv', index=False)
