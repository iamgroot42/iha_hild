"""
    Use adversarial examples to generate signal for membershup.
    Approach #1: Use standard norm-constrained attack and take note of number of iterations taken.
    Approach #2: Use standard norm-constrained attack and take note of perturbation norm.
    Approach #3: Use attack that explicitly minimizes perturbation norm and take note of norm.
"""

import torch as ch
import time
import numpy as np
from torch.autograd import Variable as V
import gc
import matplotlib.pyplot as plt

# Set high DPI also
plt.rcParams["figure.dpi"] = 250

from bbeval.attacker.transfer_methods._manipulate_input import (
    clip_by_tensor,
    transformation_function,
)
from bbeval.config import AttackerConfig, ExperimentConfig
from bbeval.attacker.transfer_methods.SMIMIFGSM import SMIMIFGSM

from mib.train import get_model_and_criterion, load_data
from tqdm import tqdm

from autoattack import AutoAttack
from superdeepfool.attacks.SuperDeepFool import SuperDeepFool


class CustomSMIMIFGSM(SMIMIFGSM):
    """
    Modified version of SMIMIFGSM from bbeval to also return iterations taken for success
    """

    def attack(self, x_orig, y_orig, y_target=None):
        eps = self.eps / 255.0
        targeted = self.targeted
        n_iters = self.params.n_iters

        # temporarily set these values for testing based on their original tf implementation
        x_min_val, x_max_val = 0, 1.0

        alpha = eps / n_iters
        decay = 1.0
        momentum = 0
        num_transformations = 12
        lamda = 1 / num_transformations

        # initializes the advesarial example
        # x.requires_grad = True
        adv = x_orig.clone()
        adv = adv.cuda()
        adv.requires_grad = True
        x_min = clip_by_tensor(x_orig - eps, x_min_val, x_max_val)
        x_max = clip_by_tensor(x_orig + eps, x_min_val, x_max_val)
        sum_time = 0

        # WB setting - model being attacked is same as model used to craft adversarial examples
        self.model.eval()  # Make sure model is in eval model
        self.model.zero_grad()  # Make sure no leftover gradients
        logits = self.model.forward(x_orig).detach()
        # Also compute loss
        loss_og = self.criterion(logits, y_orig).cpu().item()
        if not targeted:
            y_target = ch.argmax(logits, 1)
        elif y_target is None:
            raise ValueError("Must provide y_target for targeted attack")

        i = 0
        while self.optimization_loop_condition_satisfied(i, sum_time, n_iters):
            Gradients = []
            if adv.grad is not None:
                adv.grad.zero_()
            start_time = time.time()
            if i == 0:
                adv = clip_by_tensor(adv, x_min, x_max)
                adv = V(adv, requires_grad=True)
            grad = 0

            for t in range(num_transformations):
                adv = adv
                output = self.model.forward(transformation_function(adv, resize_to=29))
                output_clone = output.clone()
                loss = self.criterion(output_clone, y_target)
                loss.backward()

                Gradients.append(adv.grad.data)

            for gradient in Gradients:
                grad += lamda * gradient

            grad = momentum * decay + grad / ch.mean(
                ch.abs(grad), dim=(1, 2, 3), keepdim=True
            )
            momentum = grad

            if targeted == True:
                adv = adv - alpha * ch.sign(grad)
            else:
                adv = adv + alpha * ch.sign(grad)
            adv = clip_by_tensor(adv, x_min, x_max)
            adv = V(adv, requires_grad=True)

            end_time = time.time()
            sum_time += end_time - start_time
            # outputs the transferability
            self.model.zero_grad()  # Make sure no leftover gradients
            target_model_output = self.model.forward(adv)
            target_model_prediction = ch.max(target_model_output, 1).indices

            if targeted:
                num_transfered = ch.count_nonzero(target_model_prediction == y_target)
            else:
                num_transfered = ch.count_nonzero(target_model_prediction != y_target)

            del output, output_clone, target_model_output, target_model_prediction
            ch.cuda.empty_cache()
            del loss
            gc.collect()  # Explicitly call the garbage collector

            i += 1

            # Break if we succeeded
            if num_transfered.item() == len(y_target):
                break

        # We don't really care about the example itself, just the difference (in l_inf norm)
        perturbation_norm = ch.norm(adv.detach() - x_orig, p=float("inf")).item()
        return perturbation_norm, i, loss_og



def main():
    currdir = "."
    currdir = "/u/as9rw/work/auditing_mi/"
    config = ExperimentConfig.load(currdir + "smimifgsm.json", drop_extra_fields=False)

    # Load target model
    model_dict = ch.load("target_model/1.pt")
    # Extract member information and model
    model_weights = model_dict["model"]
    model, _ = get_model_and_criterion("cifar10", device="cpu")
    model.load_state_dict(model_weights, strict=False)
    # Shift to CUDA
    model = model.cuda()
    # Make sure it's on eval model
    model.eval()

    train_index, test_index = model_dict["train_index"], model_dict["test_index"]

    # Get data
    all_data = load_data(None, None)
    # Get indices out of range(len(all_data)) that are not train_index or test_index
    other_indices = np.array(
        [
            i
            for i in range(len(all_data))
            if i not in train_index and i not in test_index
        ]
    )

    # CIFAR
    num_train_points = 25000
    num_samples_test = 500

    # Create Subset datasets for members
    member_dset = ch.utils.data.Subset(all_data, train_index)
    # and non- members
    nonmember_indices = np.random.choice(other_indices, num_train_points, replace=False)
    nonmember_dset = ch.utils.data.Subset(all_data, nonmember_indices)

    # Attacker
    attacker_config: AttackerConfig = config.first_attack_config()
    attacker = CustomSMIMIFGSM(
        model, aux_models=None, config=attacker_config, experiment_config=config
    )

    def collect_signal(z):
        x_ = z[0].unsqueeze(0).cuda()
        y_true = ch.tensor([z[1]]).cuda()
        # Use given label as y_
        y_ = y_true
        # Use predicted label as y_
        # y_ = ch.argmax(model(x_), 1)
        # Use hardest label as y_
        # y_ = ch.argmin(model(x_), 1)
        return aa_pert_dist(model, x_, y_, y_true)
        # Also compute loss, while at it
        return attacker.attack(x_, y_true, y_target=y_)

    signals_in, signals_out = [], []
    # Collect member signals
    for mem in tqdm(
        member_dset, total=num_samples_test, desc="Collecting member signals"
    ):
        signals_in.append(collect_signal(mem))
        if len(signals_in) >= num_samples_test:
            break

    # Collect non-member signals
    for nonmem in tqdm(
        nonmember_dset, total=num_samples_test, desc="Collecting non-member signals"
    ):
        signals_out.append(collect_signal(nonmem))
        if len(signals_out) >= num_samples_test:
            break

    signals_in = np.array(signals_in)
    signals_out = np.array(signals_out)
    np.save("signals_adv_true_apgd.npy", {"in": signals_in, "out": signals_out})
    # np.save("signals_adv_targ_smimi.npy", {"in": signals_in, "out": signals_out})

    plt.scatter(signals_in[:, 0], signals_in[:, 1], alpha=0.75, label="in")
    plt.scatter(signals_out[:, 0], signals_out[:, 1], alpha=0.75, label="out")
    plt.legend()
    plt.savefig("adv_targ_smimi.png")
    exit(0)

    # Set min x-lim to 0
    plt.xlim(left=0)
    plt.hist(signals_in[:, 1], alpha=0.75, label="in", bins=20)
    plt.hist(signals_out[:1], alpha=0.75, label="out", bins=20)
    plt.legend()
    plt.savefig("adv_sdf.png")


if __name__ == "__main__":
    main()
