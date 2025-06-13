import os
import sys
import time
import logging
import functools
from pathlib import Path
import wandb

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from torch.optim.lr_scheduler import LambdaLR


from dataset_synthetic import ALIKEDSyntheticDataset
from train_wrapper import ALIKEDTrainWrapper
from torch.utils.data import random_split




logger = logging.getLogger(__name__)


class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """

    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.autograd.set_detect_anomaly(True)

    # Initialize WandB
    wandb.init(project="alikeds_synthetic", entity="ivan-nikolov")

    # Define dataset parameters
    poses_csv = "PATH_TO_DATASET/image-matching-challenge-2024-duplicated/train/train_labels.csv"
    root_dir = "PATH_TO_DATASET/image-matching-challenge-2024-duplicated/train"
    
    image_size = (640, 640)
    warp_strength = 0.1
    augment = True

    # Create dataset instance
    dataset = ALIKEDSyntheticDataset(poses_csv, root_dir, image_size, warp_strength, augment)

    # Log dataset information
    logging.info(f"Dataset contains {len(dataset)} images.")
    
    # Optionally, you can log some sample images to WandB
    sample = dataset[0]

    wandb_logger = WandbLogger(project="alikeds_synthetic", entity="ivan-nikolov")
    wandb.log({"sample_image": [wandb.Image(sample['image0'].numpy().transpose(1, 2, 0), caption="Image 0"),
                                wandb.Image(sample['image1'].numpy().transpose(1, 2, 0), caption="Image 1")]})

   

    accumulate_grad_batches = 6
    batch_size = 2

    lr_scheduler = functools.partial(WarmupConstantSchedule, warmup_steps=10)

    model = ALIKEDTrainWrapper(weights='./imc24lightglue/weights/aliked-n16.pth')
    # Split dataset into train and validation sets (e.g., 80% train, 20% val)
    val_split = 0.2
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    print(f"Total dataset size: {total_size}, Train size: {train_size}, Val size: {val_size}")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logging.info(f"Train set: {len(train_dataset)} images, Val set: {len(val_dataset)} images.")

    train_datalodader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='auto',
        accumulate_grad_batches=accumulate_grad_batches,
        logger=[wandb_logger, CSVLogger(save_dir='logs', name='alikeds_synthetic')],
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=1),
            pl.callbacks.LearningRateMonitor(logging_interval='step')
        ]
    )
    trainer.fit(
        model,
        train_dataloaders=train_datalodader,
        val_dataloaders=val_dataloader
    )