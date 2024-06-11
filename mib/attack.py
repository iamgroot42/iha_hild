"""
    File for generating signal corresponding to a given attack.
    Scores can be read later from the saved file.
"""
import argparse
import os
import torch as ch
import numpy as np
from tqdm import tqdm
import dill
import multiprocessing
import torch.multiprocessing as mp

from mib.models.utils import get_model
from sklearn.metrics import roc_curve, auc
from mib.dataset.utils import get_dataset
from mib.attacks.utils import get_attack
from mib.utils import get_signals_path, get_models_path, get_misc_path
from mib.train import get_loader
from sklearn.ensemble import RandomForestClassifier

"""
# Deterministic
ch.use_deterministic_algorithms(True)
ch.backends.cudnn.deterministic = True
ch.backends.cudnn.benchmark = False
"""


class DillProcess(mp.Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target = dill.dumps(
            self._target
        )  # Save the target function as bytes, using dill

    def run(self):
        if self._target:
            self._target = dill.loads(
                self._target
            )  # Unpickle the target function before executing
            self._target(*self._args, **self._kwargs)  # Execute the target function


def member_nonmember_loaders(
    train_data,
    train_idx,
    num_points_sample: int,
    args,
    num_nontrain_pool: int = None,
    batch_size: int = 1,
    want_all_member_nonmember: bool = False
):
    """
    num_points_sample = -1 means all points should be used, and # of non-members will be set to be = # of members
    """
    other_indices_train = np.array(
        [i for i in range(len(train_data)) if i not in train_idx]
    )

    want_all_members_and_equal_non_members = num_points_sample == -1
    if want_all_members_and_equal_non_members:
        num_points_sample = min(len(train_idx), len(other_indices_train))

    # Create Subset datasets for members
    if want_all_member_nonmember:
        train_index_subset = train_idx
    else:
        np.random.seed(args.exp_seed)
        train_index_subset = np.random.choice(train_idx, num_points_sample, replace=False)

    # Sample non-members
    np.random.seed(args.exp_seed + 1)
    nonmember_indices = np.random.choice(
        other_indices_train,
        (
            num_nontrain_pool
            if not want_all_members_and_equal_non_members
            else num_points_sample
        ),
        replace=False,
    )

    if want_all_member_nonmember or want_all_members_and_equal_non_members:
        nonmember_index_subset = nonmember_indices
        nonmember_dset_ft = None
    else:
        # Break nonmember_indices here into 2 - one for sprinkling in FT data, other for actual non-members
        nonmember_indices_ft = nonmember_indices[: num_nontrain_pool // 2]
        nonmember_indices_test = nonmember_indices[num_nontrain_pool // 2 :]

        nonmember_dset_ft = ch.utils.data.Subset(train_data, nonmember_indices_ft)

        # Sample non-members
        np.random.seed(args.exp_seed + 2)
        nonmember_index_subset = np.random.choice(
            nonmember_indices_test,
            num_points_sample,
            replace=False,
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

    # Assert no overlap between train_index_subset and nonmember_index_subset
    # Just to make sure nothing went wrong above!
    if len(set(train_index_subset).intersection(set(nonmember_index_subset))) != 0:
        print("Non-overlap found between train and non-member data. Shouldn't have happened! Check code.")
        exit(0)

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


def get_signals(return_dict,
                args,
                attacker, loader, ds,
                is_train: bool,
                nonmember_dset_ft,
                model_dir: str,
                model_map: np.ndarray,
                learning_rate: float,
                num_samples: int,
                ref_models = None,):
    # Weird multiprocessing bug when using a dill wrapper, so need to re-import
    from tqdm import tqdm

    # Compute signals for member data
    signals = []
    for i, (x, y) in tqdm(enumerate(loader), total=len(loader)):
        out_models_use = None
        in_models_use = None
        if attacker.reference_based:
            in_models_use = [ref_models[j] for j in np.nonzero(model_map[:, i])[0]]
            if args.num_ref_models is not None:
                in_models_use = in_models_use[: args.num_ref_models]

            # For L-mode, load out models specific to datapoint
            if args.l_mode and is_train:
                this_dir = os.path.join(model_dir, f"l_mode/{i}")
                out_models_use, ref_indices = load_ref_models(this_dir, args)
            else:
                # Use existing ref models
                out_models_use = [
                    ref_models[j] for j in np.nonzero(1 - model_map[:, i])[0][:]
                ]
                if args.num_ref_models is not None:
                    out_models_use = out_models_use[: args.num_ref_models]
                    in_models_use = in_models_use[: args.num_ref_models]

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
            x_aug=x_aug,
            learning_rate=learning_rate,
            num_samples=num_samples,
            is_train=is_train,
        )
        signals.append(score)
    return_dict["member" if is_train else "nonmember"] = signals


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
        num_nontrain_pool
    )

    # Temporary (for Hessian-related attack)
    entire_train_data_loader = get_loader(train_data, train_index, 512, num_workers=0)

    hessian = None
    # If attack uses uses_hessian, try loading from disk if available
    hessian_store_path = os.path.join(
        get_misc_path(), args.dataset, args.model_arch, str(args.target_model_index)
    )
    if os.path.exists(os.path.join(hessian_store_path, "hessian.ch")):
        hessian = ch.load(os.path.join(hessian_store_path, "hessian.ch"))
        print("Loaded Hessian!")

    # For reference-based attacks, train out models
    attacker_mem = get_attack(args.attack)(
        target_model,
        criterion,
        all_train_loader=entire_train_data_loader,
        hessian=hessian,
        damping_eps=args.damping_eps,
        low_rank=args.low_rank,
        approximate=args.approximate_ihvp,
        device="cuda:0",
    )

    attacker_nonmem = get_attack(args.attack)(
        target_model,
        criterion,
        all_train_loader=entire_train_data_loader,
        hessian=hessian,
        damping_eps=args.damping_eps,
        low_rank=args.low_rank,
        approximate=args.approximate_ihvp,
        device="cuda:1",
    )

    """
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
    """

    if attacker_mem.reference_based and not args.l_mode:
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

        """
        # Compute traces, if required
        if attacker_mem.requires_trace:
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
        """
    else:
        member_map = None
        nonmember_map = None
        ref_models = None
        # Shift model to CUDA (won't have ref-based models in memory as well)
        target_model.cuda()

    # Shared dict to get reutnr values
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []

    # Process for member data
    p = DillProcess(
        target=get_signals,
        args=(
            return_dict,
            args,
            attacker_mem,
            member_loader,
            ds,
            True,
            nonmember_dset_ft,
            model_dir,
            member_map,
            learning_rate,
            num_samples,
            ref_models,
        ),
    )
    p.start()
    processes.append(p)
    # Process for non-member data
    p = DillProcess(
        target=get_signals,
        args=(
            return_dict,
            args,
            attacker_nonmem,
            nonmember_loader,
            ds,
            False,
            nonmember_dset_ft,
            model_dir,
            nonmember_map,
            learning_rate,
            num_samples,
            ref_models,
        ),
    )
    p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Extract relevant data
    signals_in = return_dict["member"]
    signals_out = return_dict["nonmember"]

    """
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
        if attacker.reference_based:
            # TODO: train out models for L-mode for non-members
            out_models_use = [ref_models[j] for j in np.nonzero(1 - nonmember_map[:, i])[0]]
            in_models_use  = [ref_models[j] for j in np.nonzero(nonmember_map[:, i])[0]]
            if args.num_ref_models is not None:
                out_models_use = out_models_use[: args.num_ref_models]
                in_models_use = in_models_use[: args.num_ref_models]

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
            x_aug=x_aug,
            learning_rate=learning_rate,
            num_samples=num_samples,
            is_train=False
        )
        signals_out.append(score)
    """

    # Save signals
    signals_in  = np.array(signals_in).flatten()
    signals_out = np.array(signals_out).flatten()
    signals_dir = get_signals_path()
    save_dir = os.path.join(signals_dir, args.dataset, args.model_arch, str(args.target_model_index))

    # Make sure save_dir exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    attack_name = args.attack
    suffix = ""
    if args.same_seed_ref:
        attack_name += "_same_seed_ref"
    if args.num_ref_models is not None:
        attack_name += f"_{args.num_ref_models}_ref"
    if args.aug:
        attack_name += "_aug"
    if attacker_mem.uses_hessian:
        attack_name += f"_damping_{args.damping_eps}_lowrank_{args.low_rank}"

    if args.sif_proper_mode:
        attack_name += "_sif_proper_mode"
    if args.suffix is not None:
        suffix = f"_{args.suffix}"
    if args.simulate_metaclf:
        attack_name += "_metaclf"

    if args.simulate_metaclf:
        # Use a 2-depth decision tree to fit a meta-classifier
        signals_in_, signals_out_ = [], []
        for i in tqdm(range(len(signals_in)), desc="Meta-Clf(In)"):
            # Concatenate all data except signals_in[i], along with signals_out
            # With appropriate labels
            X = np.concatenate((signals_in[:i], signals_in[i + 1 :], signals_out)).reshape(-1, 1)
            Y = np.concatenate((np.ones(len(signals_in) - 1), np.zeros(len(signals_out))))
            # Use randomforest
            clf = RandomForestClassifier(max_depth=2)
            clf.fit(X, Y)
            # Get prediction for signals_in[i]
            pred = clf.predict_proba(signals_in[i].reshape(1, -1))[0][1]
            signals_in_.append(pred)
        # Repeat same for signals_out
        for i in tqdm(range(len(signals_out)), desc="Meta-Clf(Out)"):
            X = np.concatenate((signals_out[:i], signals_out[i + 1 :], signals_in)).reshape(-1, 1)
            Y = np.concatenate((np.zeros(len(signals_out) - 1), np.ones(len(signals_in))))
            clf = RandomForestClassifier(max_depth=2)
            clf.fit(X, Y)
            pred = clf.predict_proba(signals_out[i].reshape(1, -1))[0][1]
            signals_out_.append(pred)
        # Replace with these scores
        signals_in = np.array(signals_in_)
        signals_out = np.array(signals_out_)

    # Print out ROC
    total_labels = [0] * len(signals_out) + [1] * len(signals_in)
    total_preds = np.concatenate((signals_out, signals_in))

    # If SIF attack, need to perform thresholding
    # Consider extreme case of actual work, where we consider n-1 setting
    if args.attack == "SIF" and args.sif_proper_mode:
        total_preds_ = []

        labels_ = np.array([0] * (len(signals_out) - 1) + [1] * len(signals_in))
        for i, (x, y) in tqdm(enumerate(nonmember_loader), total=len(nonmember_loader)):
            logit = target_model(x.cuda())
            if type(criterion).__name__ == "MSELoss":
                pred = (logit.squeeze(1) > 0.5).float().cpu()
            else:
                pred = logit.argmax(dim=1).cpu()
            if pred != y:
                total_preds_.append(0.0)
            else:
                scores_ = np.concatenate(
                    (signals_out[:i], signals_out[i + 1 :], signals_in)
                )
                min_t, max_t = attacker_nonmem.get_thresholds(scores_, labels_)
                if min_t < signals_out[i] and signals_out[i] < max_t:
                    total_preds_.append(1.0)
                else:
                    total_preds_.append(0.0)

        # Repeat for member target data
        labels_ = np.array([0] * len(signals_out) + [1] * (len(signals_in) - 1))
        for i, (x, y) in tqdm(enumerate(member_loader), total=len(member_loader)):
            logit = target_model(x.cuda())
            # If MSE loss
            if type(criterion).__name__ == "MSELoss":
                pred = (logit.squeeze(1) > 0.5).float().cpu()
            else:
                pred = logit.argmax(dim=1).cpu()
            if pred != y:
                total_preds_.append(0.)
            else:
                scores_ = np.concatenate((signals_out, signals_in[:i], signals_in[i + 1 :]))
                min_t, max_t = attacker_mem.get_thresholds(scores_, labels_)
                if min_t < signals_in[i] and signals_in[i] < max_t:
                    total_preds_.append(1.)
                else:
                    total_preds_.append(0.)

        # Proceed to use these scores for MIA
        # Replace total_preds
        total_preds = np.array(total_preds_)
        # Because these scores are 0/1, TPR/FPR will be 0/1 as well
        # Choose to focus on AUC instead

    fpr, tpr, _ = roc_curve(total_labels, total_preds)
    roc_auc = auc(fpr, tpr)
    print("AUC: %.3f" % roc_auc)

    # Save results
    np.save(
        f"{save_dir}/{attack_name}{suffix}.npy",
        {
            "in": signals_in,
            "out": signals_out,
        },
    )

    # Save Hessian, if computed and didn't exist before
    if attacker_mem.uses_hessian and (
        not os.path.exists(os.path.join(hessian_store_path, "hessian.ch"))
    ):
        os.makedirs(hessian_store_path, exist_ok=True)
        ch.save(
            attacker_mem.get_hessian(), os.path.join(hessian_store_path, "hessian.ch")
        )
        print("Saved Hessian!")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_arch", type=str, default="wide_resnet_28_2")
    args.add_argument("--dataset", type=str, default="cifar10")
    args.add_argument("--attack", type=str, default="LOSS")
    args.add_argument("--exp_seed", type=int, default=2024)
    args.add_argument("--damping_eps", type=float, default=2e-1, help="Damping for Hessian computation (only valid for some attacks)")
    args.add_argument(
        "--approximate_ihvp",
        action="store_true",
        help="If true, use approximate iHV (using CG) instead of exact iHVP",
    )
    args.add_argument(
        "--low_rank",
        action="store_true",
        help="If true, use low-rank approximation of Hessian. Else, use damping. Useful for inverse",
    )
    args.add_argument("--target_model_index", type=int, default=0)
    args.add_argument("--num_ref_models", type=int, default=None)
    args.add_argument(
        "--simulate_metaclf",
        action="store_true",
        help="If true, use scores as features and fit LOO-style meta-classifier for each target datapoint",
    )
    args.add_argument("--l_mode", action="store_true", help="L-mode (where out reference model is trained on all data except target record)")
    args.add_argument("--aug", action="store_true", help="Use augmented data?")
    args.add_argument("--sif_proper_mode", action="store_true", help="Tune 2 thresholds for SIF like original paper")
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

    mp.set_start_method('spawn')
    main(args)
