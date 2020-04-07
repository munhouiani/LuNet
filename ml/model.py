from argparse import Namespace

import torch
from pytorch_lightning import LightningModule
from torch import nn as nn
from torch.nn import functional as F


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

        if hparams.out == 2:  # binary classification
            out_dim = 1
        else:
            out_dim = hparams.out
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
