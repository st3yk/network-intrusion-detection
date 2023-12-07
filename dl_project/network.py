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

    def _get_input_size(self, input_dim, layer):
        temp_x = torch.randn(1, 1, input_dim, requires_grad=False)
        temp_x = self.conv(temp_x)
        temp_x = self.max_pool(temp_x)
        if layer == "batch_norm":
            return temp_x.shape[1]
        elif layer == "lstm":
            return temp_x.shape[2]

    # Network layers
    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x, hidden = self.lstm(x)
        x = self.dropout(x)
        x = self.avg_pool(x)
        return x

    def training_step(self, batch, batch_index):
        print(f"{batch}, size: {batch[0].size()}")
        x = batch["feature"].float()
        y_hat = self(x)

        y = batch["attack_cat"].long()
        loss = {"loss": F.cross_entropy(y_hat, y)}
        if (batch_index % 50) == 0:
            self.logger.log_metrics(loss)
        return loss

    def validation_step(self, batch, batch_index):
        x = batch["feature"].float()
        y_hat = self(x)

        y = batch["attack_cat"].long()
        loss = {"val_loss": F.cross_entropy(y_hat, y)}
        if (batch_index % 50) == 0:
            self.logger.log_metrics(loss)
        return loss

    def test_step(self):
        raise NotImplemented

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.001)
