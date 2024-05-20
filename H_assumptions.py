"""
    Empirical evaluation of assumptions made over H_0 and H_1. Uses LOO-setting models to evaluate the assumptions.
"""
import argparse
import os
import torch as ch
import numpy as np
from tqdm import tqdm

from mib.models.utils import get_model
from mib.dataset.utils import get_dataset
from mib.attacks.utils import get_attack
from mib.utils import get_models_path
from mib.attacks.theory import compute_trace
from mib.train import get_loader
from mib.attack import member_nonmember_loaders
from mib.attacks.theory_new import compute_hessian, fast_ihvp, ProperTheoryRef


import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "Times New Roman"


def get_specific_grad(model, criterion, point_x, point_y):
    model.zero_grad()
    logits = model(point_x.cuda())
    loss = criterion(logits, point_y.unsqueeze(0).float().cuda())
    loss.backward()
    flat_grad = []
    for p in model.parameters():
        flat_grad.append(p.grad.detach().view(-1))
    flat_grad = ch.cat(flat_grad)
    model.zero_grad()
    return flat_grad


def main(args):
    model_dir = os.path.join(get_models_path(), args.dataset, args.model_arch)

    ds = get_dataset(args.dataset)(augment=False)
    approximate = False

    points_per_model = 1000
    eps = 1e-2
    ds = get_dataset("mnistodd")()

    train_data = ds.get_train_data()

    model_dir_1 = os.path.join(model_dir, f"replica/{args.target_split}/")
    # Load up all of these models that essentially correspond to training on D
    H_1, ihvps_1 = [], []
    random_gradients = []
    for mpath in tqdm(os.listdir(model_dir_1)):
        # Load target model
        model_1, criterion, hparams = get_model(args.model_arch, ds.num_classes)
        learning_rate = hparams["learning_rate"]
        model_dict = ch.load(os.path.join(model_dir_1, mpath))
        train_index = model_dict["train_index"]
        model_1.load_state_dict(model_dict["model"], strict=False)
        model_1.eval()
        model_1.cuda()
        train_loader = get_loader(train_data, train_index, 512)

        # Compute gradients for N random datapoints
        r_random_points = np.random.choice(len(train_index), points_per_model)
        for r in r_random_points:
            x, y = train_data[train_index[r]]
            random_gradients.append(get_specific_grad(model_1, criterion, x.cuda(), y.cuda()))

        # TODO: Generate gradients for N randomly-selected datapoints
        if approximate:
            # Directly compute iHVPs
            raise NotImplementedError("Not implemented yet!")
        else:
            exact_H = compute_hessian(model_1, train_loader, criterion, device="cuda")
            L, Q = ch.linalg.eigh(exact_H.cpu().clone().detach())
            H_1.append((L, Q))

        model_1.cpu()

    # Now we focus on models trained on D \ {z} (LOO setting) for various 'z'
    model_dir_0 = os.path.join(model_dir, f"l_mode/{args.target_split}/")
    for i_dp, lod in enumerate(os.listdir(model_dir_0)):
        H_0 = []
        # Load models trained with D \ {z} for a particular z
        model_dir_z_0 = os.path.join(model_dir_0, lod)
        for mpath in tqdm(os.listdir(model_dir_z_0), total=len(H_1)):
            # Load target model
            model_0, criterion, _ = get_model(args.model_arch, ds.num_classes)
            model_dict = ch.load(os.path.join(model_dir_z_0, mpath))
            train_index = model_dict["train_index"]
            model_0.load_state_dict(model_dict["model"], strict=False)
            model_0.eval()
            train_loader = get_loader(train_data, train_index, 512)

            if approximate:
                # Directly compute iHVPs
                raise NotImplementedError("Not implemented yet!")
            else:
                # TODO: Compute Hessian
                exact_H = compute_hessian(model_0, train_loader, criterion, device="cuda")
                L, Q = ch.linalg.eigh(exact_H.cpu().clone().detach())
                H_0.append((L, Q))

            # Stop when len(H_1) == len(H_0)
            if len(H_0) == len(H_1):
                break

        if not approximate:
            # Compute log ratio sum of eigenvalues and iHVPs
            ratios, cosine, l2 = [], [], []
            for h1 in H_1:
                H1_inv = h1[1] @ ch.diag(1 / (h1[0] + eps)) @ h1[1].t()
                ihvps_1 = ch.stack([H1_inv @ g.cpu() for g in random_gradients])
                for h0 in H_0:
                    eig0 = h0[0]
                    eig1 = h1[0]
                    # Only consider positive eigenvalues
                    eig0 = eig0[eig0 > 0]
                    eig1 = eig1[eig1 > 0]
                    min_len = min(len(eig0), len(eig1))
                    eig0 = eig0[:min_len]
                    eig1 = eig1[:min(len(eig0), len(eig1))]
                    ratio_top    = ch.sum(ch.log(2 - learning_rate * eig1))
                    ratio_bottom = ch.sum(ch.log(2 - learning_rate * eig0))
                    log_ratio_sum = ratio_top - ratio_bottom
                    ratios.append(log_ratio_sum.item())

                    # Also compute iHVPs
                    H0_inv = h0[1] @ ch.diag(1 / (h0[0] + eps)) @ h0[1].t()
                    ihvps_0 = ch.stack([H0_inv @ g.cpu() for g in random_gradients])
                    # Compute pair-wise cosine similarity and L2 norms
                    cosine.append(ch.cosine_similarity(ihvps_0, ihvps_1, dim=1).mean().item())
                    l2.append(ch.norm(ihvps_0 - ihvps_1, dim=1, p=2).mean().item())

            # Log-sum ratio
            print(len(ratios), "pair-wise values")
            sns.histplot(ratios, kde=True, stat="probability")
            plt.savefig(f"log_ratio_sum_{i_dp}.png")
            plt.clf()
            # Cosine similarity
            sns.histplot(cosine, kde=True, stat="probability")
            plt.savefig(f"cosine_sim_{i_dp}.png")
            plt.clf()
            # L2 norm difference
            sns.histplot(l2, kde=True, stat="probability")
            plt.savefig(f"l2_diff_{i_dp}.png")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_arch", type=str, default="lr_mse")
    args.add_argument("--dataset", type=str, default="mnistodd")
    args.add_argument("--target_split", type=int, default=0)
    args.add_argument(
        "--same_seed_ref",
        action="store_true",
        help="Use ref models with same seed as target model?",
    )
    args = args.parse_args()
    main(args)
