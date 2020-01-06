"""
Copyright (c) 2020 CRISP

data generator

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np

import utils

def train_ae(
    net,
    data_loader,
    hyp,
    criterion,
    optimizer,
    scheduler,
    PATH="",
    test_loader=None,
    epoch_start=0,
    epoch_end=1,
):

    info_period = hyp["info_period"]
    noiseSTD = hyp["noiseSTD"]
    device = hyp["device"]
    zero_mean_filters = hyp["zero_mean_filters"]
    normalize = hyp["normalize"]
    network = hyp["network"]
    mu = hyp["mu"]
    supervised = hyp["supervised"]
    Jg = hyp["num_trials"]
    data_distribution = hyp["data_distribution"]
    peak = hyp["peak"]

    if normalize:
        net.normalize()

    if hyp["denoising"]:
        if test_loader is not None:
            with torch.no_grad():
                psnr = []
                for idx_test, (img_test, _) in enumerate(test_loader):
                    img_test = img_test.to(device)
                    if data_distribution == "binomial":
                        sampler = torch.distributions.bernoulli.Bernoulli(probs=img_test)
                        img_test_noisy = sampler.sample()
                        for j in range(Jg-1):
                            img_test_noisy += sampler.sample()
                        img_test_noisy /= Jg
                    elif data_distribution == "poisson":
                        Q = torch.max(img_test) / peak
                        rate = img_test / Q
                        if torch.isnan(torch.min(rate)):
                            continue
                        sampler = torch.distributions.poisson.Poisson(rate)
                        img_test_noisy = sampler.sample() * Q


                    Hx_hat, _, _ = net(img_test_noisy, mu)

                    if data_distribution == "binomial":
                        img_test_hat = torch.nn.Sigmoid()(Hx_hat + mu)
                    elif data_distribution == "poisson":
                        img_test_hat = torch.exp(Hx_hat + mu)

                    psnr.append(
                        utils.PSNR(
                            img_test[0, 0, :, :].detach().cpu().numpy(),
                            img_test_hat[0, 0, :, :].detach().cpu().numpy(),
                        )
                    )

                    noisy_psnr = utils.PSNR(
                        img_test[0, 0, :, :].detach().cpu().numpy(),
                        img_test_noisy[0, 0, :, :].detach().cpu().numpy(),
                    )

                np.save(os.path.join(PATH, "psnr_init.npy"), np.array(psnr))
                print("PSNR: input {}, output {}".format(np.round(np.array(noisy_psnr), decimals=4), np.round(np.array(psnr), decimals=4)))

    for epoch in tqdm(range(epoch_start, epoch_end)):
        scheduler.step()
        loss_all = 0
        for idx, (img, _) in tqdm(enumerate(data_loader)):
            optimizer.zero_grad()

            img = img.to(device)
            if data_distribution == "binomial":
                sampler = torch.distributions.bernoulli.Bernoulli(probs=img)
                img_noisy = sampler.sample()
                for j in range(Jg-1):
                    img_noisy += sampler.sample()
                img_noisy /= Jg
            elif data_distribution == "poisson":
                Q = torch.max(img) / peak
                rate = img / Q
                if torch.isnan(torch.min(rate)):
                    continue
                sampler = torch.distributions.poisson.Poisson(rate)
                img_noisy = sampler.sample() * Q

            Hx, _, _ = net(img_noisy, mu)

            if supervised:
                loss = criterion(img, Hx, mu)
            else:
                loss = criterion(img_noisy, Hx, mu)

            loss_all += float(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if zero_mean_filters:
                net.zero_mean()
            if normalize:
                net.normalize()

            if idx % info_period == 0:
                print("loss:{:.4f}\n".format(loss.item()))

            torch.cuda.empty_cache()

        # ===================log========================

        if hyp["denoising"]:
            if test_loader is not None:
                with torch.no_grad():
                    psnr = []
                    for idx_test, (img_test, _) in enumerate(test_loader):
                        img_test = img_test.to(device)
                        if data_distribution == "binomial":
                            sampler = torch.distributions.bernoulli.Bernoulli(probs=img_test)
                            img_test_noisy = sampler.sample()
                            for j in range(Jg-1):
                                img_test_noisy += sampler.sample()
                            img_test_noisy /= Jg
                        elif data_distribution == "poisson":
                            Q = torch.max(img_test) / peak
                            rate = img_test / Q
                            if torch.isnan(torch.min(rate)):
                                continue
                            sampler = torch.distributions.poisson.Poisson(rate)
                            img_test_noisy = sampler.sample() * Q


                        Hx_hat, _, _ = net(img_test_noisy, mu)

                        if data_distribution == "binomial":
                            img_test_hat = torch.nn.Sigmoid()(Hx_hat + mu)
                        elif data_distribution == "poisson":
                            img_test_hat = torch.exp(Hx_hat + mu)

                        psnr.append(
                            utils.PSNR(
                                img_test[0, 0, :, :].detach().cpu().numpy(),
                                img_test_hat[0, 0, :, :].detach().cpu().numpy(),
                            )
                        )

                    np.save(
                        os.path.join(PATH, "psnr_epoch{}.npy".format(epoch)), np.array(psnr)
                    )
                    print("PSNR: {}".format(np.round(np.array(psnr), decimals=4)))

        torch.save(loss_all, os.path.join(PATH, "loss_epoch{}.pt".format(epoch)))

        torch.save(
            net.H.weight.data, os.path.join(PATH, "H_epoch{}.pt".format(epoch))
        )

        if network == "DEA2DtrainablebiasBinomial" or network == "DEA2DtrainablebiasPoisson":
            torch.save(net.relu.lam, os.path.join(PATH, "lam_epoch{}.pt".format(epoch)))

        print(
            "epoch [{}/{}], loss:{:.4f} ".format(
                epoch + 1, hyp["num_epochs"], loss.item()
            )
        )

    return net
