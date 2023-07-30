import argparse
import collections
import torch
import numpy as np
from data_loader import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from model.model import *
#################################################
# Importing catalyst for speeding up training process
from sklearn.model_selection import train_test_split
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
    path = "/home/sandesh/Downloads/MyGitHubArch/kaggle-clouds-segmentation-challenge/dataset"
    train = pd.read_csv(f'{path}/train.csv')
    sub = pd.read_csv(f'{path}/sample_submission.csv')


    train["label"] = train["Image_Label"].apply(lambda x: x.split("_")[1])
    train["im_id"] = train["Image_Label"].apply(lambda x: x.split("_")[0])

    sub["label"] = sub["Image_Label"].apply(lambda x: x.split("_")[1])
    sub["im_id"] = sub["Image_Label"].apply(lambda x: x.split("_")[0])

    DEVICE = "cuda"

    print(train)
    id_mask_count = train.loc[train["EncodedPixels"].isnull() == False, "Image_Label"].apply(lambda x:x.split("_")[0]).value_counts().reset_index().rename(columns = {"index":"img_id","Image_Label":"count"})
    print(id_mask_count)
    id_mask_count.columns = ["img_id", "count"]
    print(id_mask_count)
    

    train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42, stratify=id_mask_count['count'], test_size=0.1)
    test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values
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
    save_path = "../saved_models/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(),"../saved_models/model_smp.pth")




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='kaggle-clouds-segmentation-challenge')
    parser.add_argument('-bs', '--batch_size', default=16, type=int,
                      help='Batch Size (default: 16)')
    parser.add_argument('-epochs', '--num_epochs', default=20, type=int,
                      help='Number of Epochs(default: 20)')
    parser.add_argument('-ds_path', '--dataset_path', default="~/Downloads", type=str,
                      help='Input the path to the kaggle clouds segmentation dataset')
    args = parser.parse_args()

    train(args)
