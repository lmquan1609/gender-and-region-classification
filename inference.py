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

from utils import *
import validate

parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str)
parser.add_argument('--model-path', type=str)

if __name__ == '__main__':
    args = vars(parser.parse_args())
    params = Params(args['config_path'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_loader = datasetnormal.fetch_dataloader(
        os.path.join(params.data_dir, 'testing128mel.pkl'),
        params.batch_size, params.num_workers, inference=True
    )
    if params.model == 'densenet':
        model = DenseNet(config.NUM_CLASSES, False, is_finetuned=params.finetuned).to(device)

    model = load_checkpoint(args['model_path'], model)

    test_df = pd.read_csv(config.TEST_GT, index_col='id')
    correct_paths, incorrect_paths = [], []
    correct_labels, incorrect_labels, rectified_label = [], [], []
    correct_percentage, incorrect_percentage, rectified_percentage = [], [], []

    correct, total = 0, 0
    model.eval()

    with torch.no_grad():
        counter = 0
        for data in test_loader:
            inputs = data[0].to(device)
            target = data[1].squeeze(1).to(device)
            
            outputs = model(inputs)

            percentages = F.softmax(outputs.data, dim=1)
            max_vals, preds = torch.max(percentages, 1)
            for i in range(len(outputs.data)):
                label, pred, max_val = target.data[i], preds.data[i], max_vals.data[i]
                if pred == label:
                    correct_paths.append(test_df.index[counter])
                    correct_labels.append(config.INVERSE_MAPPINGS[int(pred.item())])
                    correct_percentage.append(f'{max_val.cpu().item():.2%}')
                else:
                    incorrect_paths.append(test_df.index[counter])
                    incorrect_labels.append(config.INVERSE_MAPPINGS[int(pred.item())])
                    rectified_label.append(config.INVERSE_MAPPINGS[int(label.item())])
                    incorrect_percentage.append(f'{max_val.cpu().item():.2%}')
                    rectified_percentage.append(f'{percentages[i][int(label.item())].cpu().item():.2%}')

                counter += 1
            
            total += target.size(0)
            correct += (preds == target).sum().item()
    
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