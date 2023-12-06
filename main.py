#!/bin/env python3
import numpy as np
from torch.utils.data import DataLoader
from dl_project.packetdataset import PacketDataset

if __name__ == "__main__":
    training_data = PacketDataset("datasets/UNSW_NB15_training.csv")
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
