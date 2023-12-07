#!/bin/env python3
from pytorch_lightning import Trainer
from dl_project.network import BhmthNet
from argparse import Namespace
from torch.utils.data import DataLoader
from dl_project.packetdataset import PacketDataset
import os


def train_network(input_dim, conv_out, kernel):
    path = os.environ["DATASET_DIR"]
    train_dataloader = DataLoader(
        PacketDataset(f"{path}/UNSW_NB15_training.csv"), batch_size=32, num_workers=7
    )
    test_dataloader = DataLoader(
        PacketDataset(f"{path}/UNSW_NB15_testing.csv"), batch_size=32, num_workers=7
    )
    hparams = Namespace(
        **{"input_dim": input_dim, "conv_out": conv_out, "kernel": kernel}
    )
    model = BhmthNet(hparams)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader
    )
