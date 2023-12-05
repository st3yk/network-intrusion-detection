#!/bin/env python3
import pandas as pd
import numpy as np
import torch as t
import PacketDataset
import os

print(pd.__version__)
print(np.__version__)

dataset_dir = os.environ["DATASET_DIR"]
training_set = PacketDataset.PacketDataset(f"{dataset_dir}/UNSW_NB15_training.csv")

batch_size = 4

trainloader = t.utils.data.DataLoader(
    training_set, batch_size=batch_size, shuffle=True, num_workers=2
)
