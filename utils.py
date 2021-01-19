import json
import os
import torch
from torch import nn
import shutil
import h5py
import numpy as np
import random
from config import config

def chunk(arr, chunk_len):
    for i in range(0, len(arr), chunk_len):
        yield arr[i: i + chunk_len]

def get_test_label(filename, test_df):
    # filename = filename[:4 + filename.find('.')]
    label_str = '_'.join([map_fn[idx] for map_fn, idx in \
                          zip([config.GENDER_MAPPINGS, config.ACCENT_MAPPINGS], \
                              list(test_df.loc[filename]))])
    return config.MAPPINGS[label_str]

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

class RunningAverage:
    def __init__(self):
        self.total = 0
        self.steps = 0
    
    def update(self, loss):
        self.total += loss
        self.steps += 1
    
    def __call__(self):
        return self.total / float(self.steps)

class HDF5DatasetWriter:
    def __init__(self, length, output_path, duration, sampling_rate, buf_size=1000):
        if os.path.isfile(output_path):
            os.remove(output_path)
            
        self.db = h5py.File(output_path, 'w')
        self.audio = self.db.create_dataset('audio', (length, duration * sampling_rate), dtype='float')
        self.values = self.db.create_dataset('values', tuple([length] + config.INPUT_DIMS), dtype='float')
        self.target = self.db.create_dataset('target', (length,), dtype='int')

        self.buf_size = buf_size
        self.buffer = {
            'audio': [],
            'values': [],
            'target': []
        }
        self.idx = 0

    def add(self, audios, values, labels):
        self.buffer['audio'].append(audios)
        self.buffer['values'].append(values)
        self.buffer['target'].append(labels)

        if len(self.buffer['audio']) >= self.buf_size:
            self.flush()
    
    def flush(self):
        next_idx = self.idx + len(self.buffer['audio'])
        self.audio[self.idx:next_idx] = self.buffer['audio']
        self.values[self.idx:next_idx] = self.buffer['values']
        self.target[self.idx:next_idx] = self.buffer['target']
        self.idx = next_idx
        self.buffer = {
            'audio': [],
            'values': [],
            'target': []
        }

    def close(self):
        if len(self.buffer['audio']) > 0:
            self.flush()
        self.db.close()

def save_checkpoint(state, is_best, epoch, checkpoint):
    filename = os.path.join(checkpoint, f'last_{epoch}.pth.tar')
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint, f'model_best_{epoch}.pth.tar'))

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

def initialize_weights(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Linear') != -1:
        nn.init.ones_(m.weight.data)

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
        # test_sound = np.pad(test_sound, (N - test_sound.shape[0])//2, mode = 'constant')
    return audio_signal.astype('float')

