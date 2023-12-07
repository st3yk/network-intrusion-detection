#!/bin/env python3

import torch
from torch import nn as nn
from pytorch_lightning import LightningModule


class BhmthNet(LightningModule):
    def __init__(self, train, test):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=hparams.conv_out, kernel_size=hparams.kernel
            ),
            nn.ReLU(),
        )
        self.max_pool = nn.Sequential(nn.MaxPool1d(kernel_size=2), nn.ReLU())
        self.batch_norm = nn.BatchNorm1d(self._get_input_size("batch_norm"))
        self.lstm = nn.LSTM(self._get_input_size("lstm"))
        self.dropout = nn.Dropout(p=0.5)

    def _get_input_size(self, layer):
        temp_x = torch.randn(1, 1, hparams.input_dim, requires_grad=False)
        temp_x = self.conv(dummy_x)
        temp_x = self.max_pool(dummy_x)
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
        x = self.globalaveragepooling(x)
        return x

    # 6 Sections from Ligthning Module
    def training_step(self, batch, batch_idx):
        x = batch["feature"].float()
        y_hat = self(x)

        y = batch["attack_cat"].long()
        loss = {"loss": F.cross_entropy(y_hat, y)}
        if (batch_idx % 50) == 0:
            self.logger.log_metrics(loss)
        return loss

    def validation_step(self):
        x = batch["feature"].float()
        y_hat = self(x)

        y = batch["attack_cat"].long()
        loss = {"val_loss": F.cross_entropy(y_hat, y)}
        if (batch_idx % 50) == 0:
            self.logger.log_metrics(loss)
        return loss

    def test_step(self):
        raise NotImplemented

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.001)
