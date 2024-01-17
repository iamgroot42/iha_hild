"""
    File for generating signal corresponding to a given attack.
    Scores can be read later from the saved file.
"""
import argparse
import os
import torch as ch
import numpy as np
from tqdm import tqdm

from mib.models.utils import get_model
from mib.dataset.utils import get_dataset
from mib.attacks.utils import get_attack


def main(args):
    # Load target model
    target_model = get_model(args.model_arch)
    model_dict = ch.load(f"./models/{args.target_model_index}.pt")
    target_model.load_state_dict(model_dict["model"], strict=False)
    target_model.eval()

    # Pick records (out of all train) to test
    train_index = model_dict["train_index"]

    # TODO: Get these stats from a dataset class
    ds = get_dataset(args.dataset)()
    # CIFAR
    num_train_points = 10000

    # Get data
    train_data = ds.get_train_data()
    # Get indices out of range(len(train_data)) that are not train_index or test_index
    other_indices_train = np.array(
        [i for i in range(len(train_data)) if i not in train_index]
    )

    # Create Subset datasets for members
    np.random.seed(args.exp_seed)
    train_index_subset = np.random.choice(train_index, args.num_points, replace=False)

    member_dset = ch.utils.data.Subset(train_data, train_index_subset)

    # Sample members
    np.random.seed(args.exp_seed + 1)
    nonmember_indices = np.random.choice(
        other_indices_train, num_train_points, replace=False
    )

    # Break nonmember_indices here into 2 - one for sprinkling in FT data, other for actual non-members
    nonmember_indices_ft = nonmember_indices[: num_train_points // 2]
    nonmember_indices_test = nonmember_indices[num_train_points // 2 :]

    nonmember_dset_ft = ch.utils.data.Subset(train_data, nonmember_indices_ft)

    # Sample non-members
    np.random.seed(args.exp_seed + 2)
    nonmember_index_subset = np.random.choice(
        nonmember_indices_test, args.num_points, replace=False
    )
    nonmember_dset = ch.utils.data.Subset(
        train_data,
        nonmember_index_subset,
    )

    # Make loader out of non-mem (no shuffle)
    member_loader = ch.utils.data.DataLoader(member_dset, batch_size=1, shuffle=False)
    nonmember_loader = ch.utils.data.DataLoader(
        nonmember_dset, batch_size=1, shuffle=False
    )

    # For reference-based attacks, train out models
    attacker = get_attack(args.attack)(target_model)

    if attacker.reference_based:
        out_models, out_indices = [], []
        if args.same_seed_ref:
            # Look specifically inside folder corresponding to this model's seed
            for m in os.listdir(f"./same_seed_models/{args.target_model_index}"):
                model = get_model(args.model_arch)
                state_dict = ch.load(
                    f"./same_seed_models/{args.target_model_index}/{m}"
                )
                train_index = state_dict["train_index"]
                model.load_state_dict(state_dict["model"], strict=False)
                model.eval()
                out_models.append(model)
                out_indices.append(train_index)
        else:
            # Load all other models in directory that are not the target model
            out_models = []
            for m in os.listdir("./models"):
                if int(m.split(".pt")[0]) != args.target_model_index:
                    model = get_model(args.model_arch)
                    state_dict = ch.load(f"./models/{m}")
                    train_index = state_dict["train_index"]
                    model.load_state_dict(state_dict["model"], strict=False)
                    model.eval()
                    out_models.append(model)
                    out_indices.append(train_index)

        # For each reference model, look at out_indices and create a 'isin' based 2D bool-map
        # Using train_index_subset
        out_indices = np.array(out_indices, dtype=object)
        member_map = np.zeros((len(out_models), len(train_index_subset)), dtype=bool)
        nonmember_map = np.zeros(
            (len(out_models), len(nonmember_index_subset)), dtype=bool
        )
        for i, out_index in enumerate(out_indices):
            member_map[i] = np.isin(train_index_subset, out_index)
            nonmember_map[i] = np.isin(nonmember_index_subset, out_index)

    else:
        # Shift model to CUDA (won't have ref-based models in memory as well)
        target_model.cuda()

    # Compute signals for member data
    signals_in, signals_out = [], []
    i = 0
    for x, y in tqdm(member_loader):
        out_models_use = None
        if attacker.reference_based:
            out_models_use = [out_models[j] for j in np.nonzero(member_map[:, i])[0]]

        score = attacker.compute_scores(
            x, y,
            out_models=out_models_use,
            other_data_source=nonmember_dset_ft,
        )
        signals_in.append(score)
        i += 1

    # Compute signals for non-member data
    i = 0
    for x, y in tqdm(nonmember_loader):
        out_models_use = None
        if attacker.reference_based:
            out_models_use = [out_models[j] for j in np.nonzero(nonmember_map[:, i])[0]]

        score = attacker.compute_scores(
            x, y,
            out_models=out_models_use,
            other_data_source=nonmember_dset_ft,
        )
        signals_out.append(score)
        i += 1

    # Save signals
    signals_in = np.concatenate(signals_in, 0)
    signals_out = np.concatenate(signals_out, 0)
    save_dir = f"/u/as9rw/work/auditing_mi/signals/{args.target_model_index}"

    # Make sure save_dir exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(
        f"{save_dir}/{args.attack}.npy",
        {
            "in": signals_in,
            "out": signals_out,
        },
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_arch", type=str, default="wide_resnet_cifar")
    args.add_argument("--dataset", type=str, default="cifar10")
    args.add_argument("--attack", type=str, default="LOSS")
    args.add_argument("--exp_seed", type=int, default=2024)
    args.add_argument("--target_model_index", type=int, default=0)
    args.add_argument(
        "--num_points",
        type=int,
        default=500,
        help="Number of samples (in and out each) to use for computing signals",
    )
    args.add_argument(
        "--same_seed_ref",
        action="store_true",
        help="Use ref models with same seed as target model?",
    )
    args = args.parse_args()
    main(args)
