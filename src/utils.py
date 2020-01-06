"""
Copyright (c) 2020 CRISP

utils

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
import numpy as np


class LogisticLoss(torch.nn.Module):
    def __init__(self, type):
        super(LogisticLoss, self).__init__()

        self.type = type

    def forward(self, y, Hx, mu):
        if self.type == "binomial":
            loss = -torch.mean(y * (Hx + mu), dim=(-1, -2)) + torch.mean(
                torch.log1p(torch.exp(Hx + mu)), dim=(-1, -2)
            )
        elif self.type == "poisson":
            loss = -torch.mean(y * (Hx + mu), dim=(-1, -2)) + torch.mean(
                torch.exp(Hx + mu), dim=(-1, -2)
            )
        return torch.mean(loss)


def normalize1d(x):
    return F.normalize(x, dim=-1)


def normalize2d(x):
    return F.normalize(x, dim=(-1, -2))


def err1d_H(H, H_hat):

    H = H.detach().cpu().numpy()
    H_hat = H_hat.detach().cpu().numpy()

    num_conv = H.shape[0]

    err = []
    for conv in range(num_conv):
        corr = np.sum(H[conv, 0, :] * H_hat[conv, 0, :])
        err.append(np.sqrt(1 - corr ** 2))
    return err


def err2d_H(H, H_hat):

    H = H.detach().cpu().numpy()
    H_hat = H_hat.detach().cpu().numpy()

    num_conv = H.shape[0]

    err = []

    for conv in range(num_conv):
        corr = np.sum(H[conv, 0, :, :] * H_hat[conv, 0, :, :])
        err.append(np.sqrt(1 - corr ** 2))
    return err


def PSNR(x, x_hat):
    mse = np.mean((x - x_hat) ** 2)
    max_x = np.max(x)
    return 20 * np.log10(max_x) - 10 * np.log10(mse)


def calc_pad_sizes(x, dictionary_dim=8, stride=1):
    left_pad = stride
    right_pad = (
        0
        if (x.shape[3] + left_pad - dictionary_dim) % stride == 0
        else stride - ((x.shape[3] + left_pad - dictionary_dim) % stride)
    )
    top_pad = stride
    bot_pad = (
        0
        if (x.shape[2] + top_pad - dictionary_dim) % stride == 0
        else stride - ((x.shape[2] + top_pad - dictionary_dim) % stride)
    )
    right_pad += stride
    bot_pad += stride
    return left_pad, right_pad, top_pad, bot_pad
