from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import numpy as np
import pickle5 as pickle
import torch
from PIL import Image

class AudioDataset(Dataset):
    def __init__(self, pickle_dir, transforms=None):
        self.data = []
        self.transforms = transforms
        self.data = pickle.loads(open(pickle_dir, 'rb').read())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        output_data = {}
        values = torch.Tensor(entry['values'])
        if self.transforms:
            values = self.transforms(values)
        target = torch.LongTensor([entry['target']])
        return values, target

def fetch_dataloader(pickle_dir, batch_size, num_workers, inference=False):
    dataset = AudioDataset(pickle_dir)
    dataloader = DataLoader(dataset, shuffle=not inference, batch_size=batch_size, num_workers=num_workers)
    return dataloader