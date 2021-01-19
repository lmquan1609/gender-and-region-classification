from torch.utils.data import Dataset, DataLoader
import torchvision
import torchaudio
import pandas as pd
import numpy as np
import pickle5 as pickle
import torch
import random
import librosa
from PIL import Image

class MelSpectrogram:
    def __init__(self, bins, mode):
        self.window_length = [25, 50, 100]
        self.hop_length = [10, 25, 50]
        self.fft = 1600
        self.melbins = bins
        self.mode = mode
        self.sr = 16000
        self.length = 250

    def __call__(self, value):
        sample = value
        limits = ((-2, 2), (.9, 1.2))

        if self.mode == 'train':
            pitch_shift = np.random.randint(limits[0][1], 1 + limits[0][1])
            time_stretch = np.random.random() * (limits[1][1] - limits[1][0]) + limits[1][0]
            new_audio = librosa.effects.time_stretch(
                librosa.effects.pitch_shift(sample, self.sr, pitch_shift), time_stretch
            )
        else:
            pitch_shift = 0
            time_stretch = 1
            new_audio = sample
        
        specs = []
        for i in range(len(self.window_length)):
            clip = torch.Tensor(new_audio)

            window_length = int(round(self.window_length[i] * self.sr / 1000))
            hop_length = int(round(self.hop_length[i] * self.sr / 1000))

            spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=self.fft, win_length=window_length,
            hop_length=hop_length, n_mels=128
            )(clip)

            eps = 1e-6
            spec = spec.numpy()
            spec = np.log(spec + eps)
            spec = np.asarray(torchvision.transforms.Resize((128, 250))(Image.fromarray(spec)))
            specs.append(spec)

        return torch.Tensor(np.array(specs))

class AudioDataset(Dataset):
    def __init__(self, pickle_dir, transforms=None):
        self.data = []
        self.transforms = transforms
        self.data = pickle.loads(open(pickle_dir, 'rb').read())

    def __len__(self):
        if self.transforms.mode == 'train':
            return 2 * len(self.data)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if idx < len(self.data):
            entry = self.data[idx]
            values = torch.Tensor(entry['values'])
        else:
            new_idx = idx - len(self.data)
            entry = self.data[new_idx]
            if self.transforms:
                values = self.transforms(entry['audio'])

        target = torch.LongTensor([entry['target']])
        return values, target

def fetch_dataloader(pickle_dir, batch_size, num_workers, mode):
    transforms = MelSpectrogram(128, mode)
    dataset = AudioDataset(pickle_dir, transforms=transforms)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    return dataloader