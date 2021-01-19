import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import librosa
import numpy as np
from models import DenseNet
import json
import argparse
import subprocess
import os
from preprocessing_h5 import extract_spectrogram
from utils import random_segment, Params, load_checkpoint
from config import config

parser = argparse.ArgumentParser()
parser.add_argument('--audio-path', type=str)
parser.add_argument('--model-path', type=str)
parser.add_argument('--config-path', type=str)
parser.add_argument('--sampling-rate', default=16000, type=int)
parser.add_argument('--duration', default=3, type=int)

def convert_rate(audio_path, dst_path):
    subprocess.call(['ffmpeg', '-loglevel', 'panic', '-i',  audio_path, 
            '-acodec', 'pcm_s16le', '-ac', '1', '-ar', str(args['sampling_rate']), dst_path])

if __name__ == '__main__':
    args = vars(parser.parse_args())
    params = Params(args['config_path'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    audio_path = os.path.abspath(args['audio_path'])
    # ext = audio_path[audio_path.rfind('.') + 1:]
    # if ext != 'wav':
    prefix = audio_path[:audio_path.rfind(os.path.sep)]
    filename = 'standard_' + audio_path[audio_path.rfind(os.path.sep) + 1:audio_path.rfind('.')] + '.wav'
    wav_path = os.path.join(prefix, filename)

    convert_rate(audio_path, wav_path)
    # print(audio_path)
    audio, sr = librosa.load(wav_path, sr=args['sampling_rate'])
    audio_segmented = random_segment(audio, args['duration'] * args['sampling_rate'])

    _, audio_feature, _ = extract_spectrogram(audio_segmented, None, args['sampling_rate'])
    audio_feature = torch.Tensor(audio_feature).unsqueeze(0)

    # load model
    if params.model == 'densenet':
        model = DenseNet(config.NUM_CLASSES, False, is_finetuned=params.finetuned).to(device)

    model = load_checkpoint(args['model_path'], model)

    # predict
    model.eval()

    with torch.no_grad():
        inputs = audio_feature.to(device)
        outputs = model(inputs)
        for label, pred in zip(config.MAPPINGS.keys(), F.softmax(outputs.data.cpu()).numpy()[0]):
            print(f'{label}:\t{pred:.2%}')