# from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM, AveragePooling2D
# from keras.layers import Dropout, Dense, TimeDistributed
# from keras.callbacks import ModelCheckpoint, CSVLogger
from python_speech_features import mfcc, logfbank
# from keras.utils import to_categorical
# from keras.models import Sequential
from scipy.io import wavfile
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os

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

def check_data():
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model.'.format(config.mode))
        with open(config.p_path,'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

def build_rand_feat():
    # tmp = check_data()
    # if tmp:
    #     return tmp.data[0], tmp.data[1]
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        rate, wav = wavfile.read('engine2/clean/'+file)
        label = df.at[file,'label']
        if wav.shape[0] <= 1600: continue
        rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = logfbank(sample,rate,nfilt=26,nfft=1103)
        # X_sample =  mfcc(sample,rate,numcep=config.nfeat,nfilt=config.nfilt, nfft=config.nfft) #original
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample)
        y.append(classes.index(label))
    config.min = _min
    config.max = _max
    X, y = np.array(X), np.array(y)
    X = (X - _min)/(_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
    y = tf.keras.utils.to_categorical(y,num_classes=3)
    config.data = (X,y)
    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
    return X, y

def get_conv_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', strides=(1,1), padding='same', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', strides=(1,1), padding='same'),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', strides=(1,1), padding='same'),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', strides=(1,1), padding='same'),
        tf.keras.layers.MaxPool2D((2,2), strides=(2, 2), padding='same'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model


df = pd.read_csv('engine2/engine3.csv') #change here
df.set_index('fname', inplace=True) #change here

for f in df.index:
    rate, signal = wavfile.read('engine2/clean/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

n_samples = 2 * int(df['length'].sum()/0.1)
prob_dist = class_dist/class_dist.sum()
choices = np.random.choice(class_dist.index,p=prob_dist)

config = Config(mode='conv')
X, y = build_rand_feat()
y_flat = np.argmax(y,axis = 1)
input_shape = (X.shape[1], X.shape[2], 1)
model = get_conv_model()
model.summary()
    

checkpoint = tf.keras.callbacks.ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode='max', 
                                                 save_best_only=True, save_weights_only=False, period=1)
csv_path = os.path.join(os.getcwd(), 'engine2/pMEngine.csv')
csvlogger = tf.keras.callbacks.CSVLogger(csv_path, separator=',', append=True)
model.fit(X,y, epochs=10, batch_size=1, shuffle = True, validation_split=0.1, callbacks=[checkpoint, csvlogger])
model.save(config.model_path)
#tf.keras.experimental.export_saved_model(model, 'engine2/path_to_saved_model')
#new_model = keras.experimental.load_from_saved_model('path_to_saved_model')
