"""
Copyright (c) 2020 CRISP

config

:author: Bahareh Tolooshams
"""

import torch

from sacred import Experiment, Ingredient

config_ingredient = Ingredient("cfg")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


@config_ingredient.config
def cfg():
    hyp = {
        "experiment_name": "default",
        "dataset": "VOC",
        "network": "DEA2DtrainablebiasPoisson",
        "num_trials": 1,
        "data_distribution": "poisson",
        "dictionary_dim": 8,
        "stride": 7,
        "num_conv": 64,
        "peak": 4,
        "L": 50,
        "num_iters": 30,
        "twosided": True,
        "batch_size": 1,
        "num_epochs": 400,
        "zero_mean_filters": False,
        "normalize": True,
        "lr": 0.5,
        "lr_decay": 0.8,
        "lr_step": 10,
        "cyclic": False,
        "amsgrad": False,
        "info_period": 2000,
        "sigma": 0.1,
        "lam": 0.1,
        "shuffle": True,
        "crop_dim": (128, 128),
        "init_with_DCT": True,
        "init_with_saved_file": False,
        "test_path": "../data/test_img/",
        "denoising": True,
        "mu": 0.0,
        "supervised": True,
        "device": device,
    }

@config_ingredient.named_config
def poisson_voc():
    hyp = {
        "experiment_name": "poisson_voc",
        "dataset": "VOC",
        "network": "DEA2DtrainablebiasPoisson",
        "num_trials": 1,
        "data_distribution": "poisson",
        "dictionary_dim": 8,
        "stride": 7,
        "num_conv": 64,
        "peak": 4,
        "L": 50,
        "num_iters": 30,
        "twosided": True,
        "batch_size": 1,
        "num_epochs": 400,
        "zero_mean_filters": False,
        "normalize": True,
        "lr": 0.5,
        "lr_decay": 0.8,
        "lr_step": 10,
        "cyclic": False,
        "amsgrad": False,
        "info_period": 2000,
        "sigma": 0.1,
        "lam": 0.1,
        "shuffle": True,
        "crop_dim": (128, 128),
        "init_with_DCT": True,
        "init_with_saved_file": False,
        "test_path": "../data/test_img/",
        "denoising": True,
        "mu": 0.0,
        "supervised": True,
    }
