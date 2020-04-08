from argparse import Namespace

import torch
from pytorch_lightning import LightningModule
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ml.dataset import UNSWNB15Dataset


class LuNetBlock(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=hparams.conv_out,
                kernel_size=hparams.kernel
            ),
            nn.ReLU()
        )
        self.max_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU()
        )
        # calc the output size after max_pool
        dummy_x = torch.randn(1, 1, hparams.input_dim, requires_grad=False)
        dummy_x = self.conv(dummy_x)
        dummy_x = self.max_pool(dummy_x)
        max_pool_out = dummy_x.shape[1]
        lstm_in = dummy_x.shape[2]

        self.batch_norm = nn.BatchNorm1d(max_pool_out)

        self.lstm = nn.LSTM(lstm_in, hparams.conv_out)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.batch_norm(x)

        # lstm and relu
        x, hidden = self.lstm(x)
        x = F.relu(x)

        # reshape
        x = x.view(x.shape[0], 1, -1)

        # drop out
        x = self.dropout(x)

        return x


class LuNet(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # config
        self.train_data_path = hparams.train_data_path
        self.val_data_path = hparams.val_data_path
        self.out_dim = hparams.out

        hparams_lu_block_1 = Namespace(**{
            'input_dim': hparams.input_dim,
            'conv_out': hparams.c1_out,
            'kernel': hparams.c1_kernel
        })
        self.lu_block_1 = LuNetBlock(hparams_lu_block_1)

        # use dummy to calc output
        dummy_x = torch.randn(1, 1, hparams.input_dim, requires_grad=False)
        dummy_x = self.lu_block_1(dummy_x)

        hparams_lu_block_2 = Namespace(**{
            'input_dim': dummy_x.shape[2],
            'conv_out': hparams.c2_out,
            'kernel': hparams.c2_kernel
        })
        self.lu_block_2 = LuNetBlock(hparams_lu_block_2)

        dummy_x = self.lu_block_2(dummy_x)

        hparams_lu_block_3 = Namespace(**{
            'input_dim': dummy_x.shape[2],
            'conv_out': hparams.c3_out,
            'kernel': hparams.c3_kernel
        })
        self.lu_block_3 = LuNetBlock(hparams_lu_block_3)

        dummy_x = self.lu_block_3(dummy_x)

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=hparams.final_conv,
                kernel_size=hparams.final_kernel,
            )
        )

        dummy_x = self.conv(dummy_x)
        self.avg_pool = nn.AvgPool1d(kernel_size=hparams.final_conv)

        dummy_x = self.avg_pool(dummy_x)
        self.drop_out = nn.Dropout(p=0.5)

        if self.out_dim == 2:  # binary classification
            out_dim = 1
        else:
            out_dim = self.out_dim
        self.out = nn.Linear(
            in_features=dummy_x.shape[1] * dummy_x.shape[2],
            out_features=out_dim

        )

    def forward(self, x):
        x = self.lu_block_1(x)
        x = self.lu_block_2(x)
        x = self.lu_block_3(x)
        x = self.conv(x)
        x = self.avg_pool(x)
        x = self.drop_out(x)

        # reshape
        x = x.view(x.shape[0], -1)

        x = self.out(x)

        return x

    def train_dataloader(self):
        data_loader = DataLoader(UNSWNB15Dataset(self.train_data_path), batch_size=16, shuffle=True, num_workers=12)

        return data_loader

    def val_dataloader(self):
        data_loader = DataLoader(UNSWNB15Dataset(self.val_data_path), batch_size=16, shuffle=True, num_workers=12)

        return data_loader

    def training_step(self, batch, batch_idx):
        x = batch['feature'].float()
        y_hat = self(x)

        if self.out_dim == 2:  # binary classification
            y = batch['label'].float()
            loss = {'loss': F.binary_cross_entropy(y_hat, y)}
        else:
            y = batch['attack_cat'].long()
            loss = {'loss': F.cross_entropy(y_hat, y)}

        if (batch_idx % 50) == 0:
            self.logger.log_metrics(loss, step=batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['feature'].float()
        y_hat = self(x)

        if self.out_dim == 2:  # binary classification
            y = batch['label'].float()
            loss = {'val_loss': F.binary_cross_entropy(y_hat, y)}
        else:
            y = batch['attack_cat'].long()
            loss = {'val_loss': F.cross_entropy(y_hat, y)}

        if (batch_idx % 50) == 0:
            self.logger.log_metrics(loss, step=batch_idx)
        return loss

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.001)
