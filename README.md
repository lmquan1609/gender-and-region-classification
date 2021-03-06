# Voice-based Gender and Region Classification
This repository contains the demo for Gender and Region classification model

## Dependencies
```
pip install -r requirements.txt
```

## Launch the Demo
1. Download the [model](https://drive.google.com/file/d/1_IQxXXoTSsCYOPNV18FAXer0JFzCXK1T/view?usp=sharing) then put it in `UI/model`
2. Launch the Demo
```
python UI/app.py
```

## Modeling
### Melspectrogram-based model
In this model, we convert the audio to image of mel-spectrogram and use DenseNet to train our classifier

Visit branch `melspectrogram` for further details

```
git checkout melspectrogram
```

### Deep feature-based model and low level based model
In this model, we convert the audio to low level feature include time domain and freq domain, then use simple model to classify. We also use Multi-scale model to learn feature from raw signal.

Visit branch `WavemsNet` for further details
```
git checkout WavemsNet
```
