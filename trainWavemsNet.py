import argparse
import os

from model.WavemsNet import WaveMsNet
from preprocessdata.dataprocessWaveMsNet import WaveformDataset

import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tensorboardX import SummaryWriter
import tqdm
import pickle

from utils import RunningAverage, Params, save_checkpoint


def load_data(filename):
    """Load data from pickle file

    Parameters
    ----------
    filename: str
        Path to file

    Returns
    -------
    data: list or dict
        Loaded file.

    """

    return pickle.load(open(filename, "rb"), encoding='latin1')

def train(model, optimizer, train_loader, epoch, loss_fn, device):

    model.train()
    # start = time.time()

    loss_avg = RunningAverage()

    with tqdm.trange(len(train_loader), desc=f'Training dataset for epoch {epoch + 1}...') as t:
        for idx, data in enumerate(train_loader):
            inputs = data[0]
            labels = data[1].squeeze(1)
            
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.item())

            t.set_postfix(loss=f'{loss_avg():05.3f}')
            t.update()
    return loss_avg()

def test(model, valid_loader, device):
    
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            inputs = data[0]
            targets = data[1].squeeze(1)
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += (preds == targets).sum().item()
    return 100*correct/total
        

def main(model, trainPkl, validPkl, writer, params, device, optim, loss_fn=nn.CrossEntropyLoss(), scheduler=None):

    trainDataset = WaveformDataset(trainPkl)
    train_loader = DataLoader(trainDataset, batch_size=params.batch_size, shuffle=True, num_workers=2)

    validDataset = WaveformDataset(validPkl)
    valid_loader = DataLoader(validDataset, batch_size=params.valid_batch_size, shuffle=True)

    best_acc = 0.0

    for epoch in range(params.epochs):
        avg_loss = train(model, optim, train_loader, epoch, loss_fn, device)

        acc = test(model, valid_loader, device)
        acc_train = test(model, train_loader, device)
        print(f"Epoch {epoch:02d} / {params.epochs}  Loss_train: {avg_loss}, Train acc: {acc_train}, Valid acc: {acc}")
        
        is_best = acc > best_acc
        if is_best: best_acc = acc

        if scheduler: scheduler.step()

        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optim.state_dict()
        }, is_best, epoch + 1, params.checkpoint_dir)

        writer.add_scalar('training_loss', avg_loss, epoch)
        writer.add_scalar('training_acc', acc_train, epoch)
        writer.add_scalar('val_acc', acc, epoch)
    
    writer.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, type=str, help="Config file path of training")
    parser.add_argument('--traindata', required=True, type=str, help="path file of training data")
    parser.add_argument('--validdata', required=True, type=str, help="path file of validation data")
    args = vars(parser.parse_args())
    params = Params(args['config_path'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter()

    model = WaveMsNet()
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr= params.lr, momentum=params.momentum, weight_decay=params.weight_decay)

    if params.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=.1)
    else:
        scheduler = None

    trainPkl = args['traindata']
    validPkl = args['validdata']
    main(model, trainPkl, validPkl, writer, params, device, optimizer, loss_fn, scheduler)