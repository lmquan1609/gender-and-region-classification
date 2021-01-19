import librosa
import argparse
import numpy as np
import pickle as pickle
import torch
import torchaudio
import torchvision
from PIL import Image
from config import config
import os
import pandas as pd
from imutils import paths
import pickle
from sklearn.model_selection import train_test_split
import tqdm
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str)
parser.add_argument('--store-dir', type=str)
parser.add_argument('--sampling-rate', default=16000, type=int)

def extract_spectrogram(values, audio, target):
    num_channels = 3
    window_sizes = [25, 50, 100]
    hop_sizes = [10, 25, 50]

    specs = []
    for i in range(num_channels):
        window_length = int(round(window_sizes[i] * args['sampling_rate'] / 1000))
        hop_length = int(round(hop_sizes[i] * args['sampling_rate'] / 1000))

        audio = torch.Tensor(audio)
        spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=args['sampling_rate'], n_fft=1600, win_length=window_length,
            hop_length=hop_length, n_mels=128
        )(audio)
        eps = 1e-6
        spec = spec.numpy()
        spec = np.log(spec + eps)
        spec = np.asarray(torchvision.transforms.Resize((128, 250))(Image.fromarray(spec)))
        specs.append(spec)

    new_entry = {}
    new_entry['audio'] = audio.numpy()
    new_entry['values'] = np.array(specs)
    new_entry['target'] = target
    values.append(new_entry)

def extract_features(audio_paths, labels, d_type):
    values = []
    for i in tqdm.trange(len(labels), desc=f'Preprocessing for {d_type}...'):
        audio_path, label = audio_paths[i], labels[i]
        clip, sr = librosa.load(audio_path, sr=args['sampling_rate'])
        extract_spectrogram(values, clip, label)
    return values

if __name__ == '__main__':
    args = vars(parser.parse_args())
    root_dir = args['data_dir']

    if not os.path.isdir(args['store_dir']): os.makedirs(args['store_dir'])

    training_audio_paths, training_labels, test_audio_paths, test_labels = [], [], [], []

    for label in config.LABELS:
        src_folder = os.path.join(root_dir, label)
        for full_path in paths.list_files(src_folder):
            training_audio_paths.append(full_path)
            training_labels.append(config.MAPPINGS[label])

    training_audio_paths, val_audio_paths, training_labels, val_labels = train_test_split(
        training_audio_paths, training_labels, stratify=training_labels,\
        test_size=.2, random_state=42
    )

    src_folder = os.path.join(root_dir, 'public_test')
    test_df = pd.read_csv(config.TEST_GT, index_col='id')
    for audio_path in test_df.index:
        test_audio_paths.append(os.path.join(src_folder, audio_path[:1 + audio_path.find('.')] + 'wav'))
        test_labels.append(get_test_label(audio_path, test_df))

    training_values = extract_features(training_audio_paths, training_labels, 'training')
    val_values = extract_features(val_audio_paths, val_labels, 'validation')
    test_values = extract_features(test_audio_paths, test_labels, 'testing')

    for d_type, values in zip(['training', 'validation', 'testing'],\
                        [training_values, val_values, test_values]):
        pickle_path = os.path.join(args['store_dir'], f'{d_type}128mel.pkl')
        f = open(pickle_path, 'wb')
        f.write(pickle.dumps(values, protocol=pickle.HIGHEST_PROTOCOL))