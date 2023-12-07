#!/bin/env python3

import torch
from torch import nn as nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule


class BhmthNet(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=hparams.conv_out, kernel_size=hparams.kernel
            ),
            nn.ReLU(),
        )
        self.max_pool = nn.Sequential(nn.MaxPool1d(kernel_size=2), nn.ReLU())
        self.batch_norm = nn.BatchNorm1d(
            self._get_input_size(hparams.input_dim, "batch_norm")
        )
        self.lstm = nn.LSTM(
            self._get_input_size(hparams.input_dim, "lstm"), hparams.conv_out
        )
        self.dropout = nn.Dropout(p=0.5)
        self.avg_pool = nn.AvgPool1d(kernel_size=hparams.conv_out)

        self.categories_dict = {
            "Normal": 0.0,
            "Fuzzers": 0.1,
            "Analysis": 0.2,
            "Backdoors": 0.3,
            "DoS": 0.4,
            "Exploits": 0.5,
            "Generic": 0.6,
            "Reconnaissance": 0.7,
            "Shellcode": 0.8,
            "Worms": 0.9,
        }

    def _get_input_size(self, input_dim, layer):
        temp_x = torch.randn(1, 1, input_dim, requires_grad=False)
        temp_x = self.conv(temp_x)
        temp_x = self.max_pool(temp_x)
        if layer == "batch_norm":
            return temp_x.shape[2]
        elif layer == "lstm":
            temp_x = self.batch_norm(temp_x)
            # temp_x = F.relu(temp_x)
            return temp_x.shape[2]

    # Network layers
    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x, hidden = self.lstm(x)
        x = self.dropout(x)
        x = self.avg_pool(x)
        return x

    def training_step(self, batch, batch_index):
        # print("--------------")
        # print(f"batch: {batch}, size: {batch[0].size()}")
        # print("--------------")
        x = batch[0].float()
        y_hat = self(x)
        y = batch[1]
        print(y.size())
        print(y_hat.size())
        loss = {"loss": F.cross_entropy(y_hat, y)}
        if (batch_index % 50) == 0:
            self.logger.log_metrics(loss)
        return loss

    def validation_step(self, batch, batch_index):
        x = batch[0].float()
        y_hat = self(x)

        y = batch[1].long()
        loss = {"val_loss": F.cross_entropy(y_hat, y)}
        if (batch_index % 50) == 0:
            self.logger.log_metrics(loss)
        return loss

    def test_step(self):
        raise NotImplemented

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.001)
