#!/bin/env python3
import numpy as np
from torch.utils.data import DataLoader
from dl_project.packetdataset import PacketDataset

if __name__ == "__main__":
    # Prepare data
    training_data = PacketDataset("datasets/UNSW_NB15_training.csv")
    testing_data = PacketDataset("datasets/UNSW_NB15_testing.csv")

    train_dataloader = DataLoader(training_data, batch_size=32)
    test_dataloader = DataLoader(testing_data, batch_size=32)
