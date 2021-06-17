# Engine RPM Classification

Simple 3 class engine rpm classification: 1000rpm, 2000rpm and 3000rpm.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install, try "pip3 install [name of package]" the following

```
python3
tensorflow
pandas
librosa
tqdm
numpy
scipy
sklearn
matplotlib
python_speech_features
```

## Running the demo

The following python files are necessary

### eda010619.py

(Optional) Visualize audio files in time domain, fft, fbank and mfcc.
Resample audio sampling rate and write into new directory. 
(Optional) function for removal of dead/silent part of audio on line 41-50. 
Execute with

```
python3 eda010619.py
```

### model010619.py

Generate logfbank/mfcc features for training and save data to pickle file. 
Comment line 50 and uncomment line 51 for mfcc feature extraction.
Custom convolutional model will be saved as well.
Execute with

```
python3 model010619.py
```

### predict010619.py

Predict on trained classes and visualize confusion metrics.
Execute with

```
python3 predict010619.py
```

### inference010619.py

Inference on individual test file.
Execute with

```
python3 inference010619.py
```