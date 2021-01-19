from torch.utils.data import Dataset, DataLoader
import torchvision
import torchaudio
import pandas as pd
import numpy as np
import pickle5 as pickle
import torch
import random
import librosa
import h5py
from .datasetaug import MelSpectrogram

class AudioDataset(Dataset):
    def __init__(self, h5_dir, transforms=None):
        self.data = h5py.File(h5_dir, 'r')
        self.transforms = transforms
    
    def __len__(self):
        if self.transforms.mode == 'train':
            return 2 * self.data['target'].shape[0]
        else:
            return self.data['target'].shape[0]

    def __getitem__(self, idx):
        if idx < self.data['target'].shape[0]:
            print(idx)
            values = torch.Tensor(self.data['values'][idx])
            target = self.data['target'][idx]
        else:
            new_idx = idx - self.data['target'].shape[0]
            print(new_idx)
            audio = self.data['audio'][new_idx]
            target = self.data['target'][new_idx]
            if self.transforms:
                values = self.transforms(audio)
        
        return values, torch.LongTensor([target])

def fetch_dataloader(h5_dir, batch_size, num_workers, mode):
    transforms = MelSpectrogram(128, mode)
    dataset = AudioDataset(h5_dir, transforms=transforms)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    return dataloader