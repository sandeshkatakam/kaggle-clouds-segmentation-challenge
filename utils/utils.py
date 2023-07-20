import json
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


import os
import cv2
import collections
import time
import tqdm
from PIL import Image
from functools import partial
train_on_gpu = True

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

###################### UTILITY AND HELPER FUNCTIONS ####################################


def plot_loss_curves(train_results, valid_results):
    train_loss = train_results['loss'].values.astype(float)
    valid_loss = valid_results['loss'].values.astype(float)
    steps = train_results['step'].values.astype(float)
    plt.plot(steps, train_loss, label='Train Loss')
    plt.plot(steps, valid_loss, label='Valid Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Validation Loss')
    plt.legend()
    plt.show()


def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    '''
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def get_image(x,folder: str = "train_images"):
  data_folder = f"{path}/{folder}"
  image_path = os.path.join(data_folder,x)
  img = cv2.imread(image_path)
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  return img

sigmoid = lambda x: 1 / (1 + np.exp(-x))

def to_tensor(x,**kwargs):
  return x.transpose(2,0,1).astype("float32")


def make_masks(df : pd.DataFrame, image_name: str = "img.jpg",shape: tuple = (1400,2100)):
  encoded_masks = df.loc[df["im_id"] == image_name,"EncodedPixels"]
  masks = np.zeros((shape[0],shape[1],4),dtype = np.float64)
  for idx, label in enumerate(encoded_masks.values):
    if label is not np.nan:
      mask = rle_decode(label)
      masks[:, :, idx] = mask

  return masks


def visualize(image, mask, original_image=None, original_mask=None):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """
    fontsize = 14
    class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(1, 5, figsize=(24, 24))

        ax[0].imshow(image)
        for i in range(4):
            ax[i + 1].imshow(mask[:, :, i])
            ax[i + 1].set_title(f'Mask {class_dict[i]}', fontsize=fontsize)
    else:
        f, ax = plt.subplots(2, 5, figsize=(24, 12))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        for i in range(4):
            ax[0, i + 1].imshow(original_mask[:, :, i])
            ax[0, i + 1].set_title(f'Original mask {class_dict[i]}', fontsize=fontsize)

        ax[1, 0].imshow(image)
        ax[1, 0].set_title('Transformed image', fontsize=fontsize)


        for i in range(4):
            ax[1, i + 1].imshow(mask[:, :, i])
            ax[1, i + 1].set_title(f'Transformed mask {class_dict[i]}', fontsize=fontsize)


def plot_with_augmentation(image,mask,augment):
  augmented = augment(image = image,mask = mask)
  image_flipped = augmented["image"]
  mask_flipped = augmented["mask"]
  visualize(image_flipped,mask_flipped,original_image = image,original_mask = mask)


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
        albu.Resize(320, 640)
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(320, 640)

    ]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]

    return albu.Compose(_transform)


def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

########################################################################################
