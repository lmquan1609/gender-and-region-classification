import torch
import torchvision
from torch import nn
import numpy as np
from models import DenseNet
from dataloaders import datasetaug, datasetnormal, datasetaug_h5, datasetnormal_h5
import json
import argparse
import tqdm

from utils import *
import validate
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str)
parser.add_argument('--checkpoint-path', type=str)
parser.add_argument('--use-h5', default=0, type=int)

def train(model, device, data_loader, optimizer, loss_fn, epoch):
    model.train()
    loss_avg = RunningAverage()

    with tqdm.trange(len(data_loader), desc=f'Training dataset for epoch {epoch + 1}...') as t:
        for batch_idx, data in enumerate(data_loader):
            inputs = data[0].to(device)
            target = data[1].squeeze(1).to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.item())

            t.set_postfix(loss=f'{loss_avg():05.3f}')
            t.update()
    
    return loss_avg()

def train_and_evaluate(model, device, train_loader, val_loader, optimizer,\
                    loss_fn, writer, params, scheduler=None, start_epoch=0):
    
    best_acc = 0.0

    for epoch in range(start_epoch, params.epochs):
        avg_loss = train(model, device, train_loader, optimizer, loss_fn, epoch)

        acc = validate.evaluate(model, device, val_loader)
        print(f"Epoch {epoch:02d} / {params.epochs}  Loss: {avg_loss}, Validation accuracy: {acc}")

        is_best = acc > best_acc
        if is_best: best_acc = acc
        if scheduler: scheduler.step()

        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None
        }, is_best, epoch + 1, params.checkpoint_dir)

        writer.add_scalar('training_loss', avg_loss, epoch)
        writer.add_scalar('val_acc', acc, epoch)
    
    writer.close()

if __name__ == '__main__':
    args = vars(parser.parse_args())
    params = Params(args['config_path'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args['use_h5'] == 0:
        aug = datasetaug
        non_aug = datasetnormal
        train_data_path = os.path.join(params.data_dir, 'training128mel.pkl')
        val_data_path = os.path.join(params.data_dir, 'validation128mel.pkl')
    else:
        aug = datasetaug_h5
        non_aug = datasetnormal_h5
        data_dir = params.h5_dir
        train_data_path = os.path.join(params.h5_dir, 'training128mel.hdf5')
        val_data_path = os.path.join(params.h5_dir, 'validation128mel.hdf5')

    if params.dataaug:
        train_loader = aug.fetch_dataloader(
            train_data_path,
            params.batch_size, params.num_workers, 'train'
        )
    else:
        train_loader = non_aug.fetch_dataloader(
            train_data_path,
            params.batch_size, params.num_workers
        )
    val_loader = non_aug.fetch_dataloader(
        val_data_path,
        params.batch_size, params.num_workers
    )

    writer = SummaryWriter()
    if params.model == 'densenet':
        model = DenseNet(config.NUM_CLASSES, params.pretrained, is_finetuned=params.finetuned).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr,\
                                weight_decay=params.weight_decay)
    
    if params.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=.1)
    else:
        scheduler = None

    start_epoch = 0

    if args['checkpoint_path']:
        model, optimizer, scheduler, start_epoch = \
            load_checkpoint(args['checkpoint_path'], model,\
                            optimizer=optimizer, scheduler=scheduler)
    
    train_and_evaluate(model, device, train_loader, val_loader, optimizer,\
                    loss_fn, writer, params, scheduler, start_epoch)