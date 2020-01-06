"""
Copyright (c) 2020 CRISP

crsae model

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
import numpy as np

import utils

use_cuda = torch.cuda.is_available()

class RELUTwosided(torch.nn.Module):
    def __init__(self, num_conv, lam=1e-3, L=100, device=None):
        super(RELUTwosided, self).__init__()
        self.L = L
        self.lam = torch.nn.Parameter(lam * torch.ones(1, num_conv, 1, 1, device=device))
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        lam = self.relu(self.lam)
        mask1 = (x > (lam / self.L)).float()
        mask2 = (x < -(lam/ self.L)).float()
        out = mask1 * (x - (lam / self.L))
        out += mask2 * (x + (lam / self.L))
        return out

class DEA1DBinomial(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(DEA1DBinomial, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.sigma = hyp["sigma"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]

        self.H = torch.nn.ConvTranspose1d(
            self.num_conv,
            1,
            kernel_size=self.dictionary_dim,
            stride=self.stride,
            bias=False,
        )
        self.HT = torch.nn.Conv1d(
            1,
            self.num_conv,
            kernel_size=self.dictionary_dim,
            stride=self.stride,
            bias=False,
        )

        if H is not None:
            self.H.weight.data = H.clone()

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.H.weight.data = self.H.weight.data.to(self.device)
        self.HT.weight.data = self.H.weight.data

    def normalize(self):
        self.H.weight.data = F.normalize(self.H.weight.data, dim=-1)
        self.HT.weight.data = self.H.weight.data

    def forward(self, x, mu):
        num_batches = x.shape[0]

        D_in = x.shape[-1]
        D_enc = D_in - self.dictionary_dim + 1

        self.lam = self.sigma * torch.sqrt(
            2 * torch.log(torch.zeros(1, device=self.device) + (self.num_conv * D_enc))
        )

        x_old = torch.zeros(num_batches, self.num_conv, D_enc, device=self.device)
        yk = torch.zeros(num_batches, self.num_conv, D_enc, device=self.device)
        x_new = torch.zeros(num_batches, self.num_conv, D_enc, device=self.device)
        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            H_yk_mu = self.H(yk) + mu
            x_tilda = x - self.sigmoid(H_yk_mu)
            x_new = yk + self.HT(x_tilda) / self.L
            if self.twosided:
                x_new = self.relu(torch.abs(x_new) - self.lam / self.L) * torch.sign(
                    x_new
                )
            else:
                x_new = self.relu(x_new - self.lam / self.L)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + (t_old - 1) / t_new * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        Hx = self.H(x_new)

        return Hx, x_new, self.lam

class DEA2DBinomial(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(DEA1DBinomial, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.sigma = hyp["sigma"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]

        self.H = torch.nn.ConvTranspose2d(
            self.num_conv,
            1,
            kernel_size=self.dictionary_dim,
            stride=self.stride,
            bias=False,
        )
        self.HT = torch.nn.Conv2d(
            1,
            self.num_conv,
            kernel_size=self.dictionary_dim,
            stride=self.stride,
            bias=False,
        )

        if H is not None:
            self.H.weight.data = H.clone()

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.H.weight.data = self.H.weight.data.to(self.device)
        self.HT.weight.data = self.H.weight.data

    def normalize(self):
        self.H.weight.data = F.normalize(self.H.weight.data, dim=(-1, -2))
        self.HT.weight.data = self.H.weight.data

    def split_image(self, x):
        if self.stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            x, self.dictionary_dim, self.stride
        )
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=self.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.stride) for j in range(self.stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(x),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x, mu):
        x_batched_padded, valids_batched = self.split_image(x)

        num_batches = x_batched_padded.shape[0]


        D_enc1 = self.HT(x_batched_padded).shape[2]
        D_enc2 = self.HT(x_batched_padded).shape[3]


        self.lam = self.sigma * torch.sqrt(
            2 * torch.log(torch.zeros(1, device=self.device) + (self.num_conv * D_enc1 * D_enc2))
        )

        x_old = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )
        yk = torch.zeros(num_batches, self.num_conv, D_enc1, D_enc2, device=self.device)
        x_new = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )

        del D_enc1
        del D_enc2
        del num_batches

        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            H_yk_mu = self.H(yk) + mu
            x_tilda = x_batched_padded - self.sigmoid(H_yk_mu)
            x_new = yk + self.HT(x_tilda) / self.L
            if self.twosided:
                x_new = (x_new > (self.lam / self.L)).float() * (x_new - (self.lam / self.L)) + (x_new < -(self.lam / self.L)).float() * (x_new + (self.lam / self.L))
            else:
                x_new = (x_new > (self.lam / self.L)).float() * (x_new - (self.lam / self.L))

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + (t_old - 1) / t_new * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = (torch.masked_select(self.H(x_new), valids_batched.byte()).reshape(
            x.shape[0], self.stride ** 2, *x.shape[1:]
        )).mean(dim=1, keepdim=False)

        return z, x_new, self.lam

class DEA2DtrainablebiasBinomial(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(DEA2DtrainablebiasBinomial, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.sigma = hyp["sigma"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]
        self.lam = hyp["lam"]

        self.H = torch.nn.ConvTranspose2d(
            self.num_conv,
            1,
            kernel_size=self.dictionary_dim,
            stride=self.stride,
            bias=False,
        )
        self.HT = torch.nn.Conv2d(
            1,
            self.num_conv,
            kernel_size=self.dictionary_dim,
            stride=self.stride,
            bias=False,
        )

        if H is not None:
            self.H.weight.data = H.clone()

        self.relu = RELUTwosided(self.num_conv, self.lam, self.L, self.device)
        self.sigmoid = torch.nn.Sigmoid()

        self.H.weight.data = self.H.weight.data.to(self.device)
        self.HT.weight.data = self.H.weight.data

    def normalize(self):
        self.H.weight.data = F.normalize(self.H.weight.data, dim=(-1, -2))
        self.HT.weight.data = self.H.weight.data

    def split_image(self, x):
        if self.stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            x, self.dictionary_dim, self.stride
        )
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=self.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.stride) for j in range(self.stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(I),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x, mu):
        x_batched_padded, valids_batched = self.split_image(x)

        num_batches = x_batched_padded.shape[0]


        D_enc1 = self.HT(x_batched_padded).shape[2]
        D_enc2 = self.HT(x_batched_padded).shape[3]


        x_old = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )
        yk = torch.zeros(num_batches, self.num_conv, D_enc1, D_enc2, device=self.device)
        x_new = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )

        del D_enc1
        del D_enc2
        del num_batches

        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            H_yk_mu = self.H(yk) + mu
            x_tilda = x_batched_padded - self.sigmoid(H_yk_mu)
            x_new = yk + self.HT(x_tilda) / self.L

            x_new = self.relu(x_new)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + (t_old - 1) / t_new * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = (torch.masked_select(self.H(x_new), valids_batched.byte()).reshape(
            x.shape[0], self.stride ** 2, *x.shape[1:]
        )).mean(dim=1, keepdim=False)


        return z, x_new, self.lam

class DEA2DtrainablebiasPoisson(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(DEA2DtrainablebiasPoisson, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.sigma = hyp["sigma"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]
        self.lam = hyp["lam"]

        self.H = torch.nn.ConvTranspose2d(
            self.num_conv,
            1,
            kernel_size=self.dictionary_dim,
            stride=self.stride,
            bias=False,
        )
        self.HT = torch.nn.Conv2d(
            1,
            self.num_conv,
            kernel_size=self.dictionary_dim,
            stride=self.stride,
            bias=False,
        )

        if H is not None:
            self.H.weight.data = H.clone()

        self.relu = RELUTwosided(self.num_conv, self.lam, self.L, self.device)

        self.H.weight.data = self.H.weight.data.to(self.device)
        self.HT.weight.data = self.H.weight.data

    def normalize(self):
        self.H.weight.data = F.normalize(self.H.weight.data, dim=(-1, -2))
        self.HT.weight.data = self.H.weight.data

    def split_image(self, x):
        if self.stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            x, self.dictionary_dim, self.stride
        )
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=self.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.stride) for j in range(self.stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(I),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x, mu):
        x_batched_padded, valids_batched = self.split_image(x)

        num_batches = x_batched_padded.shape[0]


        D_enc1 = self.HT(x_batched_padded).shape[2]
        D_enc2 = self.HT(x_batched_padded).shape[3]


        x_old = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )
        yk = torch.zeros(num_batches, self.num_conv, D_enc1, D_enc2, device=self.device)
        x_new = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )

        del D_enc1
        del D_enc2
        del num_batches

        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            H_yk_mu = self.H(yk) + mu
            x_tilda = x_batched_padded - torch.exp(H_yk_mu)
            x_new = yk + self.HT(x_tilda) / self.L

            x_new = self.relu(x_new)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + (t_old - 1) / t_new * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = (torch.masked_select(self.H(x_new), valids_batched.byte()).reshape(
            x.shape[0], self.stride ** 2, *x.shape[1:]
        )).mean(dim=1, keepdim=False)


        return z, x_new, self.lam

class DnCNN(torch.nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3, device=None):
        super(DnCNN, self).__init__()
        padding = 1
        layers = []

        layers.append(torch.nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(torch.nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(torch.nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = torch.nn.Sequential(*layers)
        self._initialize_weights()

        self.dncnn.to(device)

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
