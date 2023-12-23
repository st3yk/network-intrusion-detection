#!/bin/env python3
from dl_project.train import train_network
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":
    logger = TensorBoardLogger("model_logs", "bhmth_model")
    train_network(input_dim=43, conv_out=18, kernel=8)
