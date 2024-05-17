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
from sklearn.metrics import roc_curve, auc
from mib.dataset.utils import get_dataset
from mib.attacks.utils import get_attack
from mib.utils import get_signals_path, get_models_path
from mib.attacks.theory import compute_trace
from mib.train import get_loader

"""
# Deterministic
ch.use_deterministic_algorithms(True)
ch.backends.cudnn.deterministic = True
ch.backends.cudnn.benchmark = False
"""


def member_nonmember_loaders(
    train_data,
    train_idx,
    num_points_sample: int,
    args,
    num_nontrain_pool: int = None,
    batch_size: int = 1,
    want_all_member_nonmember: bool = False
):
    other_indices_train = np.array(
        [i for i in range(len(train_data)) if i not in train_idx]
    )

    # Create Subset datasets for members
    if want_all_member_nonmember:
        train_index_subset = train_idx
    else:
        np.random.seed(args.exp_seed)
        train_index_subset = np.random.choice(train_idx, num_points_sample, replace=False)

    # Sample non-members
    np.random.seed(args.exp_seed + 1)
    nonmember_indices = np.random.choice(
        other_indices_train, num_nontrain_pool, replace=False
    )

    if want_all_member_nonmember:
        nonmember_index_subset = nonmember_indices
    else:
        # Break nonmember_indices here into 2 - one for sprinkling in FT data, other for actual non-members
        nonmember_indices_ft = nonmember_indices[: num_nontrain_pool // 2]
        nonmember_indices_test = nonmember_indices[num_nontrain_pool // 2 :]

        nonmember_dset_ft = ch.utils.data.Subset(train_data, nonmember_indices_ft)

        # Sample non-members
        np.random.seed(args.exp_seed + 2)
        nonmember_index_subset = np.random.choice(
            nonmember_indices_test, num_points_sample, replace=False
        )

    # Make dsets
    member_dset = ch.utils.data.Subset(train_data, train_index_subset)
    nonmember_dset = ch.utils.data.Subset(
        train_data,
        nonmember_index_subset,
    )

    # Make loaders out of data
    member_loader = ch.utils.data.DataLoader(
        member_dset, batch_size=batch_size, shuffle=False
    )
    nonmember_loader = ch.utils.data.DataLoader(
        nonmember_dset, batch_size=batch_size, shuffle=False
    )

    if want_all_member_nonmember:
        return member_loader, nonmember_loader
    return (
        member_loader,
        nonmember_loader,
        nonmember_dset_ft,
        train_index_subset,
        nonmember_index_subset,
    )


def load_ref_models(model_dir, args, num_classes: int):
    if args.same_seed_ref:
        folder_to_look_in = os.path.join(model_dir, f"same_init/{args.target_model_index}")
    else:
        folder_to_look_in = model_dir

    if args.specific_ref_folder is not None:
        folder_to_look_in = os.path.join(folder_to_look_in, args.specific_ref_folder)

    # Look specifically inside folder corresponding to this model's seed
    ref_models, ref_indices = [], []
    for m in os.listdir(folder_to_look_in):
        # Skip if directory
        if os.path.isdir(os.path.join(folder_to_look_in, m)):
            continue

        # Skip ref model if trained on exact same data split as target model
        # if m.split(".pt")[0].split("_")[0] == f"{args.target_model_index}":
        #    continue

        model, _, _ = get_model(args.model_arch, num_classes)
        state_dict = ch.load(os.path.join(folder_to_look_in, m))
        ref_indices.append(state_dict["train_index"])
        model.load_state_dict(state_dict["model"], strict=False)
        model.eval()
        ref_models.append(model)

    ref_indices = np.array(ref_indices, dtype=object)
    return ref_models, ref_indices


def main(args):
    model_dir = os.path.join(get_models_path(), args.dataset, args.model_arch)

    # TODO: Get these stats from a dataset class
    ds = get_dataset(args.dataset)(augment=False)

    # Load target model
    target_model, criterion, hparams = get_model(args.model_arch, ds.num_classes)
    model_dict = ch.load(os.path.join(model_dir, f"{args.target_model_index}.pt"))
    target_model.load_state_dict(model_dict["model"], strict=False)
    target_model.eval()

    # Pick records (out of all train) to test
    train_index = model_dict["train_index"]

    # Get some information about model
    learning_rate = hparams['learning_rate']
    num_samples = len(train_index)

    # CIFAR
    num_nontrain_pool = 10000

    # Get data
    train_data = ds.get_train_data()

    (
        member_loader,
        nonmember_loader,
        nonmember_dset_ft,
        train_index_subset,
        nonmember_index_subset,
    ) = member_nonmember_loaders(
        train_data,
        train_index,
        args.num_points,
        args,
        num_nontrain_pool,
    )

    # Temporary (for Hessian-related attack)
    entire_train_data_loader = get_loader(train_data, train_index, 512)

    # For reference-based attacks, train out models
    attacker = get_attack(args.attack)(target_model, criterion, all_train_loader=entire_train_data_loader)

    # Register trace if required
    if attacker.requires_trace:
        # Compute and add trace value to model
        mem_loader_, nonmem_loader_ = member_nonmember_loaders(
            train_data,
            train_index,
            args.num_points,
            args,
            num_nontrain_pool=5000,
            want_all_member_nonmember=True,
            batch_size=256,
        )
        model_trace = compute_trace(target_model, mem_loader_, nonmem_loader_)
        # Trace business (for theory-based attack)
        attacker.register_trace(model_trace)

    if attacker.reference_based and not args.l_mode:
        ref_models, ref_indices = load_ref_models(model_dir, args, ds.num_classes)

        # For each reference model, look at ref_indices and create a 'isin' based 2D bool-map
        # Using train_index_subset
        member_map = np.zeros((len(ref_models), len(train_index_subset)), dtype=bool)
        nonmember_map = np.zeros(
            (len(ref_models), len(nonmember_index_subset)), dtype=bool
        )
        for i, out_index in enumerate(ref_indices):
            member_map[i] = np.isin(train_index_subset, out_index)
            nonmember_map[i] = np.isin(nonmember_index_subset, out_index)

        # Compute traces, if required
        if attacker.requires_trace:
            ref_traces = []
            for m, ids in tqdm(
                zip(ref_models, ref_indices), desc="Computing traces", total=len(ref_models)
            ):
                # Get in, out loaders corresponding to these models
                mem_loader, nonmem_loader = member_nonmember_loaders(
                    train_data,
                    ids,
                    args.num_points,
                    args,
                    # num_train_points=num_train_points,
                    num_nontrain_pool=5000,
                    want_all_member_nonmember=True,
                    batch_size=256,
                )
                # Get traces corresponding to these models
                ref_trace = compute_trace(m, mem_loader, nonmem_loader)
                ref_traces.append(ref_trace)
    else:
        # Shift model to CUDA (won't have ref-based models in memory as well)
        target_model.cuda()

    # Compute signals for member data
    signals_in, signals_out = [], []
    for i, (x, y) in tqdm(enumerate(member_loader), total=len(member_loader)):
        out_models_use = None
        in_models_use = None
        out_traces_use = None
        if attacker.reference_based:
            in_models_use = [ref_models[j] for j in np.nonzero(member_map[:, i])[0]]
            if args.num_ref_models is not None:
                in_models_use = in_models_use[: args.num_ref_models]

            # For L-mode, load out models specific to datapoint
            if args.l_mode:
                this_dir = os.path.join(model_dir, f"l_mode/{i}")
                out_models_use, ref_indices = load_ref_models(this_dir, args)
            else:
                # Use existing ref models
                out_models_use = [ref_models[j] for j in np.nonzero(1 - member_map[:, i])[0][:]]
                if attacker.requires_trace:
                    out_traces_use = [ref_traces[j] for j in np.nonzero(1 - member_map[:, i])[0]]
                if args.num_ref_models is not None:
                    out_models_use = out_models_use[: args.num_ref_models]
                    in_models_use = in_models_use[: args.num_ref_models]
                    if attacker.requires_trace:
                        out_traces_use = out_traces_use[: args.num_ref_models]

        # Apply input augmentations
        x_aug = None
        if args.aug:
            x_aug = ds.get_augmented_input(x, y)

        score = attacker.compute_scores(
            x,
            y,
            out_models=out_models_use,
            in_models=in_models_use,
            other_data_source=nonmember_dset_ft,
            out_traces=out_traces_use,
            x_aug=x_aug,
            learning_rate=learning_rate,
            num_samples=num_samples,
            is_train=True
        )
        signals_in.append(score)

    # Compute signals for non-member data
    for i, (x, y) in tqdm(enumerate(nonmember_loader), total=len(nonmember_loader)):
        out_models_use = None
        in_models_use = None
        out_traces_use = None
        if attacker.reference_based:
            # TODO: train out models for L-mode for non-members
            out_models_use = [ref_models[j] for j in np.nonzero(1 - nonmember_map[:, i])[0]]
            in_models_use  = [ref_models[j] for j in np.nonzero(nonmember_map[:, i])[0]]
            if attacker.requires_trace:
                out_traces_use = [ref_traces[j] for j in np.nonzero(1 - nonmember_map[:, i])[0]]
            if args.num_ref_models is not None:
                out_models_use = out_models_use[: args.num_ref_models]
                in_models_use = in_models_use[: args.num_ref_models]
                if attacker.requires_trace:
                    out_traces_use = out_traces_use[: args.num_ref_models]

        # Apply input augmentations
        x_aug = None
        if args.aug:
            x_aug = ds.get_augmented_input(x, y)

        score = attacker.compute_scores(
            x,
            y,
            out_models=out_models_use,
            in_models=in_models_use,
            other_data_source=nonmember_dset_ft,
            out_traces=out_traces_use,
            x_aug=x_aug,
            learning_rate=learning_rate,
            num_samples=num_samples,
            is_train=False
        )
        signals_out.append(score)

    # Save signals
    signals_in  = np.array(signals_in).flatten()
    signals_out = np.array(signals_out).flatten()
    signals_dir = get_signals_path()
    save_dir = os.path.join(signals_dir, args.dataset, args.model_arch, str(args.target_model_index))

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.histplot(signals_in, kde=True, stat='probability')
    sns.histplot(signals_out, kde=True, color="orange", stat='probability')
    plt.xlabel(r"$loss - (I_2 + I_3)$")
    plt.savefig("debug.png")

    # Make sure save_dir exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    attack_name = args.attack
    suffix = ""
    if args.same_seed_ref:
        attack_name += "_same_seed_ref"
    if args.num_ref_models is not None:
        attack_name += f"_{args.num_ref_models}_ref"
    if args.suffix is not None:
        suffix = f"_{args.suffix}"
    if args.aug:
        attack_name += "_aug"

    # Print out ROC
    total_labels = [0] * len(signals_out) + [1] * len(signals_in)
    total_preds = np.concatenate((signals_out, signals_in))
    fpr, tpr, _ = roc_curve(total_labels, total_preds)
    roc_auc = auc(fpr, tpr)
    print("AUC: %.3f" % roc_auc)

    np.save(
        f"{save_dir}/{attack_name}{suffix}.npy",
        {
            "in": signals_in,
            "out": signals_out,
        },
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_arch", type=str, default="wide_resnet_28_2")
    args.add_argument("--dataset", type=str, default="cifar10")
    args.add_argument("--attack", type=str, default="LOSS")
    args.add_argument("--exp_seed", type=int, default=2024)
    args.add_argument("--target_model_index", type=int, default=0)
    args.add_argument("--num_ref_models", type=int, default=None)
    args.add_argument("--l_mode", action="store_true", help="L-mode (where out reference model is trained on all data except target record)")
    args.add_argument("--aug", action="store_true", help="Use augmented data?")
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
    args.add_argument(
        "--specific_ref_folder",
        type=str,
        default=None,
        help="Custom ref sub-folder to load ref models from",
    )
    args.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Custom suffix (folder) to load models from",
    )
    args = args.parse_args()
    main(args)
