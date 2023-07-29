import segmentation_models_pytorch as smp
import catalyst
from catalyst.callbacks.metrics.segmentation import DiceCallback
from catalyst.callbacks.misc import EarlyStoppingCallback
from catalyst.callbacks.checkpoint import CheckpointCallback
from catalyst import dl
from catalyst.contrib.losses.dice import DiceLoss
from catalyst.contrib.losses.iou import IoULoss
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

ENCODER = "resnet50"
ENCODER_WEIGHTS = "imagenet"
DEVICE = "cuda"

ACTIVATION = None
model = smp.Unet(encoder_name = ENCODER,encoder_weights = ENCODER_WEIGHTS,classes = 4,activation = ACTIVATION)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER,ENCODER_WEIGHTS)

optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 1e-2},
    {'params': model.encoder.parameters(), 'lr': 1e-3},
])
scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
criterion = DiceLoss()


