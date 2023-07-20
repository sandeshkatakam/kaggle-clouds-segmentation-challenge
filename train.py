import argparse
import collections
import torch
import numpy as np
from data_loader.data_loaders import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from model.model import *
#################################################
# Importing catalyst for speeding up training process

import catalyst
from catalyst import dl
from catalyst.contrib.losses.dice import DiceLoss
from catalyst.contrib.losses.iou import IoULoss

from catalyst.runners.supervised import SupervisedRunner
from catalyst.callbacks.metrics.segmentation import DiceCallback
from catalyst.callbacks.misc import EarlyStoppingCallback
from catalyst.callbacks.checkpoint import CheckpointCallback





def train(args):
    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    num_workers = 0
    bs = args.batch_size
    train_dataset = CloudDataSet(df = train,datatype = "train",img_ids = train_ids,transforms = get_training_augmentation(), preprocessing = get_preprocessing(preprocessing_fn))
    valid_dataset = CloudDataSet(df = train,datatype = "valid",img_ids = valid_ids,transforms = get_validation_augmentation(), preprocessing = get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)
    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }
    num_epochs = args.num_epochs 
    logdir = "../logdir/segmentation/"
    runner = SupervisedRunner()
    runner.train(
        model=model,
        engine=dl.GPUEngine("cuda:0"),
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True
    )
    torch.save(model.state_dict(),"../saved_models/model_smp.pth")

## TODO: 
# test.py
# postprocess.py


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser(description='kaggle-clouds-segmentation-challenge')
    args.add_argument('-bs', '--batch_size', default=16, type=int,
                      help='Batch Size (default: 16)')
    args.add_argument('-epochs', '--num_epochs', default=20, type=str,
                      help='Number of Epochs(default: 20)')

    train(args)
