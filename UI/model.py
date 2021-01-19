import numpy as np
import librosa
import sys
import subprocess
import librosa

import random
import torch
import torchaudio
import torchvision
from torch import nn
from torchvision import models
import torch.nn.functional as F
from PIL import Image

import os
cur_dir = os.path.dirname(os.path.abspath(__file__))

label_mapping={0:'female_north',1:'female_central',2:'female_south',
                3:'male_north',4:'male_central',5:'male_south'}

# my_model=tf.keras.models.load_model(cur_dir+'model\\model.01-0.57.h5')

def random_segment(audio_signal, N):
    """Select the audio range which have duration of N
    Args:
        audio_signal (array): [description]
        N (int): duration to be extracted from original audio
    Returns:
        array: audio array having duration of N
    """    
    length = audio_signal.shape[0]
    if N < length:
        start = random.randint(0, length - N)
        audio_signal = audio_signal[start:start + N]
    else: 
        tmp = np.zeros((N,))
        start = random.randint(0, N - length)
        tmp[start: start + length] = audio_signal 
        audio_signal = tmp
    return audio_signal.astype('float')

def Convert_Data(fname):
    y, sr = librosa.load(fname)
    N = int(3 * sr)
    segment = random_segment(y, N)
    return segment.reshape(1,-1)

def predict(name):
    """predict gender and regions of accent in audio
    Args:
        name (string): path file audio
    Returns:
        y: Probability vectors of each label
        label: label predict
    """
    audio = Convert_Data(name)
    y=my_model.predict(audio)
    y=(y*100).round(2)
    label=label_mapping[np.argmax(y)]
    return y,label

class DenseNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, is_finetuned=False):
        super().__init__()
        self.model = models.densenet201(pretrained=pretrained)
        if is_finetuned:
            self.model.features.conv0.requires_grad = False
            self.model.features.norm0.requires_grad = False
            self.model.features.relu0.requires_grad = False
            self.model.features.pool0.requires_grad = False
            self.model.features.denseblock1.requires_grad = False
            self.model.features.transition1.requires_grad = False
            self.model.features.denseblock2.requires_grad = False
            self.model.features.transition2.requires_grad = False

        self.model.classifier = nn.Linear(1920, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output

def convert_rate(audio_path, dst_path):
    subprocess.call(['ffmpeg', '-y', '-loglevel', 'panic', '-i',  audio_path, 
            '-acodec', 'pcm_s16le', '-ac', '1', '-ar', str(16000), dst_path])

def extract_spectrogram(audio, target, sampling_rate):
    num_channels = 3
    window_sizes = [25, 50, 100]
    hop_sizes = [10, 25, 50]

    specs = []
    for i in range(num_channels):
        window_length = int(round(window_sizes[i] * sampling_rate / 1000))
        hop_length = int(round(hop_sizes[i] * sampling_rate / 1000))

        audio = torch.Tensor(audio)
        spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate, n_fft=1600, win_length=window_length,
            hop_length=hop_length, n_mels=128
        )(audio)
        eps = 1e-6
        spec = spec.numpy()
        spec = np.log(spec + eps)
        spec = np.asarray(torchvision.transforms.Resize((128, 250))(Image.fromarray(spec)))
        specs.append(spec)

    return audio.numpy(), np.array(specs), target

def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None, device='cpu'):
    if not os.path.exists(checkpoint):
        raise(f'File not found {checkpoint}')
    
    checkpoint = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            return model, optimizer, scheduler, start_epoch
        return model, optimizer
    return model

def predict_pytorch(name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    audio_path = name
    prefix = audio_path[:audio_path.rfind(os.path.sep)]
    filename = 'standard_' + audio_path[audio_path.rfind(os.path.sep) + 1:audio_path.rfind('.')] + '.wav'
    wav_path = os.path.join(prefix, filename)

    convert_rate(audio_path, wav_path)
    audio, sr = librosa.load(wav_path, sr=16000)
    os.remove(wav_path)
    audio_segmented = random_segment(audio, 3 * 16000)

    _, audio_feature, _ = extract_spectrogram(audio_segmented, None, 16000)
    audio_feature = torch.Tensor(audio_feature).unsqueeze(0)

    model = DenseNet(6, False, is_finetuned=False).to(device)
    model_path = os.path.join(cur_dir, 'model', 'model_best_35.pth.tar')
    model = load_checkpoint(model_path, model)

    model.eval()

    with torch.no_grad():
        inputs = audio_feature.to(device)
        outputs = model(inputs)
        
        prob = F.softmax(outputs.data.cpu()).numpy()
    return (prob*100).round(2), label_mapping[np.argmax(prob)]
