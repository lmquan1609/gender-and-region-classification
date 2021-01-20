import argparse
import os

from model.WavemsNet import WaveMsNet
from preprocessdata.dataprocessWaveMsNet import WaveformDataset
from utils import RunningAverage, save_checkpoint, Params, load_checkpoint

import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tensorboardX import SummaryWriter
import tqdm
import pickle as pickle


def test(model, testloader, device):
    
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            inputs = data[0]
            targets = data[1].squeeze(1)
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += (preds == targets).sum().item()
    return 100*correct/total
        

def main(model, testPkl, device):

    # trainDataset = WaveformDataset(trainPkl)
    # train_loader = DataLoader(trainDataset, batch_size=params.batch_size, shuffle=True, num_workers=2)

    testDataset = WaveformDataset(testPkl)
    test_loader = DataLoader(testDataset, batch_size=32, shuffle=False)

    acc = test(model, test_loader, device)

    print(f"Accuracy publictest {acc}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Model path")
    parser.add_argument('--testfile', type=str, required=True, help="Test file path")
    args = vars(parser.parse_args())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    model = WaveMsNet()
    model = load_checkpoint(args['model_path'], model)
    model = model.to(device)
    

    testPkl = args['testfile']
    main(model, testPkl, device)