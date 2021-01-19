import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import numpy as np
from models import DenseNet
from dataloaders import datasetaug, datasetnormal
import json
import argparse
import tqdm
import pandas as pd
from config import config
import librosa
from preprocessing_h5 import extract_spectrogram

from utils import *
import validate

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str)
parser.add_argument('--model-path', type=str)
parser.add_argument('--public-test', type=str)
parser.add_argument('--sampling-rate', default=16000, type=int)
parser.add_argument('--duration', default=3, type=int)
parser.add_argument('--batch-size', default=32, type=int)

if __name__ == '__main__':
    args = vars(parser.parse_args())
    params = Params(args['config_path'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_audio_paths, test_labels = [], []

    test_df = pd.read_csv(config.TEST_GT, index_col='id')
    for audio_path in test_df.index:
        test_audio_paths.append(os.path.join(args['public_test'], audio_path))
        test_labels.append(get_test_label(audio_path, test_df))

    samples, sample_labels = [], []

    correct_paths, incorrect_paths = [], []
    correct_labels, incorrect_labels, rectified_label = [], [], []
    correct_percentage, incorrect_percentage, rectified_percentage = [], [], []

    correct, total = 0, 0

    # load model
    if params.model == 'densenet':
        model = DenseNet(config.NUM_CLASSES, False, is_finetuned=params.finetuned).to(device)
    
    model = load_checkpoint(args['model_path'], model)
    model.eval()
    with torch.no_grad():
        for i in tqdm.trange(len(test_labels)):
            audio_path, label = test_audio_paths[i], test_labels[i]
            audio, sr = librosa.load(audio_path, sr=args['sampling_rate'])
            audio_segmented = random_segment(audio, args['duration'] * args['sampling_rate'])

            _, audio_feature, _ = extract_spectrogram(audio_segmented, None, args['sampling_rate'])
            samples.append(audio_feature)
            sample_labels.append(label)

            if len(samples) >= args['batch_size'] or i == len(test_labels) - 1:
                audio_samples = torch.Tensor(samples)
                inputs = audio_samples.to(device)
                outputs = model(inputs)

                percentages = F.softmax(outputs.data, dim=1)
                max_vals, preds = torch.max(percentages, 1)
                for j in range(len(outputs.data)):
                    start_idx = args['batch_size'] * (i // args['batch_size'])
                    label, pred, max_val = sample_labels[j], preds.data[j], max_vals.data[j]
                    if pred == label:
                        correct_paths.append(test_df.index[j + start_idx])
                        correct_labels.append(config.INVERSE_MAPPINGS[int(pred.item())])
                        correct_percentage.append(f'{max_val.cpu().item():.2%}')
                    else:
                        incorrect_paths.append(test_df.index[j + start_idx])
                        incorrect_labels.append(config.INVERSE_MAPPINGS[int(pred.item())])
                        rectified_label.append(config.INVERSE_MAPPINGS[int(label)])
                        incorrect_percentage.append(f'{max_val.cpu().item():.2%}')
                        rectified_percentage.append(f'{percentages[j][int(label)].cpu().item():.2%}')
                total += len(samples)
                correct += (preds == torch.Tensor(sample_labels)).sum().item()
                samples, sample_labels = [], []
    
    print(f"Validation accuracy: {100 * correct / total}")

    correct_df = pd.DataFrame({
        'label': correct_labels,
        'percentage': correct_percentage
    }, index=correct_paths)

    incorrect_df = pd.DataFrame({
        'label': rectified_label,
        'label_percentage': rectified_percentage,
        'prediction': incorrect_labels,
        'pred_percentage': incorrect_percentage,
    }, index=incorrect_paths)

    correct_df.to_csv(config.CORRECT_CLASSIFICATION_PATH)
    incorrect_df.to_csv(config.INCORRECT_CLASSIFICATION_PATH)