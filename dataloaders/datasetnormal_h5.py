from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import numpy as np
import pickle5 as pickle
import torch
from PIL import Image
import h5py

class AudioDataset(Dataset):
    def __init__(self, h5_dir, transforms=None):
        self.data = h5py.File(h5_dir, 'r')
        self.transforms = transforms
    
    def __len__(self):
        return self.data['target'].shape[0]
    
    def __getitem__(self, idx):
        values, target = self.data['values'][idx], self.data['target'][idx]
        values = torch.Tensor(values)
        target = torch.LongTensor([target])
        return values, target

def fetch_dataloader(h5_dir, batch_size, num_workers):
    dataset = AudioDataset(h5_dir)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    return dataloader