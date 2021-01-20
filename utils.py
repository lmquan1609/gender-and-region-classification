import os
import h5py
import numpy as np
import json
import shutil

import torch
from torch import nn
# from config import config

def chunk(arr, chunk_len):
    for i in range(0, len(arr), chunk_len):
        yield arr[i: i + chunk_len]

class RunningAverage:
    def __init__(self):
        self.total = 0
        self.steps = 0
    
    def update(self, loss):
        self.total += loss
        self.steps += 1
    
    def __call__(self):
        return self.total / float(self.steps)

def save_checkpoint(state, is_best, epoch, checkpoint):
    filename = os.path.join(checkpoint, f'last_{epoch}.pth.tar')
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint, f'model_best_{epoch}.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer=None, device='cpu'):
    if not os.path.exists(checkpoint):
        raise(f'File not found {checkpoint}')
    
    checkpoint = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model

class Params:
    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        f = open(json_path, 'w')
        f.write(json.dumps(self.__dict__, indent=4))
        f.close()

    def update(self, json_path):
        params = json.loads(open(json_path).read())
        self.__dict__.update(params)
    
    @property
    def dict(self):
        return self.__dict__