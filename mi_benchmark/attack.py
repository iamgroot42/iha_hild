"""
    File for generating signal corresponding to a given attack.
    Scores can be read later from the saved file.
"""
import argparse
import os
import torch as ch
import numpy as np

from mi_benchmark.models.utils import get_model


def get_data(*args):
    return


def main(args):
    # Load target model
    target_model = get_model(args.model_arch)
    model_dict = ch.load(f"./models/{args.target_model_index}.pt")["model"]
    target_model.load_state_dict(model_dict, strict=False)
    target_model.eval()

    # Pick records (out of all train) to test
    train_index = model_dict["train_index"]

    # TODO: Get these stats from a dataset class
    # CIFAR
    num_train_points = 10000

    # Get data
    train_data, _ = get_data(just_want_data=True)
    # Get indices out of range(len(train_data)) that are not train_index or test_index
    other_indices_train = np.array(
        [i for i in range(len(train_data)) if i not in train_index]
    )

    # Create Subset datasets for members
    member_dset = ch.utils.data.Subset(
        train_data, np.random.choice(train_index, args.num_points, replace=False)
    )
    # and non- members
    nonmember_indices = np.random.choice(
        other_indices_train, num_train_points, replace=False
    )
    # Break nonmember_indices here into 2 - one for sprinkling in FT data, other for actual non-members
    nonmember_indices_ft = nonmember_indices[: num_train_points // 2]
    nonmember_indices_test = nonmember_indices[num_train_points // 2 :]

    nonmember_dset_ft = ch.utils.data.Subset(train_data, nonmember_indices_ft)
    nonmember_dset = ch.utils.data.Subset(
        train_data,
        np.random.choice(nonmember_indices_test, args.num_points, replace=False),
    )

    # For reference-based attacks, train out models
    attacker = None
    out_models, out_indices = [], []
    if attacker.reference_based:
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


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_arch", type=str, default="wide_resnet_cifar")
    args.add_argument("--dataset", type=str, default="cifar10")
    args.add_argument("--target_model_index", type=int, default=0)
    args.add_argument(
        "--num_points",
        type=int,
        default=500,
        help="Number of samples (in and out each) to use for computing signals",
    )
    args.add_argument(
        "--same_seed_ref",
        type=bool,
        action="store_true",
        help="Use ref models with same seed as target model?",
    )
    args = args.parse_args()
    main(args)
