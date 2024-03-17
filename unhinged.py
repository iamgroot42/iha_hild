"""
    Use knowledge of membership from X% of actual training data to train membership meta-classifier.
    Potentially unrealistic threat model as an adversary, but plausible for auditing.
"""

# Silence annoying 'TensorRT' warnings, esp when we're not even using Tensorflow
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warning messages
# import tensorflow as tf

import os
import torch as ch
import numpy as np
import argparse
from tqdm import tqdm

from sklearn.metrics import roc_auc_score

from mib.attack import member_nonmember_loaders
from mib.models.utils import get_model
from mib.dataset.utils import get_dataset
from mib.utils import get_signals_path, get_models_path
import torch.multiprocessing as mp

from mib.attacks.meta_audit import (
    train_meta_clf,
    MetaModelCNN,
    MetaModePlain,
    MetaModelSiamese,
    FeaturesDataset,
    dict_collate_fn
)


def main(args):
    model_dir = os.path.join(get_models_path(), args.dataset, args.model_arch)

    # TODO: Get these stats from a dataset class
    ds = get_dataset(args.dataset)(augment=False)

    # Load target model
    target_model, _, _ = get_model(args.model_arch, ds.num_classes)
    model_dict = ch.load(os.path.join(model_dir, f"{args.target_model_index}.pt"))
    target_model.load_state_dict(model_dict["model"], strict=False)
    target_model.eval()

    # Pick records (out of all train) to test
    train_index = model_dict["train_index"]

    # CIFAR
    num_nontrain_pool = 25000 #25000 #10000

    # Get data
    train_data = ds.get_train_data()

    (
        member_loader,
        nonmember_loader,
    ) = member_nonmember_loaders(
        train_data,
        train_index,
        None,
        args,
        num_nontrain_pool,
        want_all_member_nonmember=True,
        batch_size=256
    )

    # For reference-based attacks, train out models
    # attacker = get_attack(args.attack)(target_model)
    target_model.cuda()

    # Collect member data
    x_in, y_in = [], []
    for i, (x, y) in tqdm(enumerate(member_loader), total=len(member_loader)):
        x_in.append(x)
        y_in.append(y)
    x_in = ch.cat(x_in, 0)
    y_in = ch.cat(y_in, 0)

    # Collect non-member data
    x_out, y_out = [], []
    for i, (x, y) in tqdm(enumerate(nonmember_loader), total=len(nonmember_loader)):
        x_out.append(x)
        y_out.append(y)
    x_out = ch.cat(x_out, 0)
    y_out = ch.cat(y_out, 0)

    # Use length of any key to get number of samples
    NUM_HOLDOUT_TEST = 500
    min_of_both = min(len(x_in), len(x_out))
    random_idx = np.random.permutation(min_of_both)
    train_idx = random_idx[NUM_HOLDOUT_TEST:]
    test_idx = random_idx[:NUM_HOLDOUT_TEST]

    # Train
    x_data_train = ch.cat((x_in[train_idx], x_out[train_idx]), 0)
    y_data_train = ch.cat((y_in[train_idx], y_out[train_idx]), 0)
    signals_train_y = np.concatenate(
        [np.ones(len(train_idx)), np.zeros(len(train_idx))]
    )

    # Test
    x_data_test = ch.cat((x_in[test_idx], x_out[test_idx]), 0)
    y_data_test = ch.cat((y_in[test_idx], y_out[test_idx]), 0)
    signals_test_y = np.concatenate(
        [np.ones(len(test_idx)), np.zeros(len(test_idx))]
    )

    # Replace with random labels
    # signals_train_y = np.random.permutation(signals_train_y)

    META_DEVICE = "cuda:1"

    # Create meta-classifier
    if args.dataset == "purchase100":
        meta_clf = MetaModePlain(
            num_classes_data=ds.num_classes, feature_mode=args.pairwise
        )
    else:
        meta_clf = MetaModelCNN(
            num_classes_data=ds.num_classes, feature_mode=args.pairwise
        )

    if args.pairwise:
        meta_clf = MetaModelSiamese(meta_clf)

    batch_size = 8 if args.pairwise else 64

    meta_clf = train_meta_clf(
        meta_clf,
        target_model,
        (x_data_train, y_data_train),
        signals_train_y,
        batch_size=batch_size,
        val_points=100,#250,
        num_epochs=args.epochs,
        device=META_DEVICE,
        augment=args.augment,
        pairwise=args.pairwise,
    )
    meta_clf.eval()

    # Clear memory
    ch.cuda.empty_cache()

    """
    # Probably easier to make a loader out of test data
    ds_test = FeaturesDataset(target_model, x_data_test, y_data_test, ch.from_numpy(signals_test_y).float(), batch_size=1)
    test_loader = ch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, collate_fn=dict_collate_fn)
    # Convert list of dicts to list of dicts
    """

    if args.augment:
        x_test_all = []
        for x in x_data_test:
            x_aug = ds.get_augmented_input(x.unsqueeze(0), None)
            x_test_all.append(x_aug)
        x_test_all = ch.stack(x_test_all, 0)
        # Swap first 2 dims (generic)
        x_test_all = x_test_all.permute(1, 0, *range(2, x_test_all.dim()))
    else:
        x_test_all = [x_data_test]

    all_preds = []
    for x_data_test in tqdm(x_test_all):
        ds_test = FeaturesDataset(target_model, x_data_test, y_data_test, ch.from_numpy(signals_test_y).float(), batch_size=16)
        test_loader = ch.utils.data.DataLoader(ds_test, batch_size=4, shuffle=False, collate_fn=dict_collate_fn)

        test_preds = []
        collected_features = []
        for batch in test_loader:
            x, y = batch[0], batch[1]
            x_ = {k: v.to(META_DEVICE) for k, v in x.items()}

            if args.pairwise:
                embed = meta_clf.get_feature_emb(x_).detach()
                collected_features.append(embed)
            else:
                preds = ch.nn.functional.sigmoid(meta_clf(x_)).detach().cpu().numpy()
                # preds = preds.mean(0)
                test_preds.extend(preds)

        if args.pairwise:
            collected_features = ch.cat(collected_features, 0)
            for i in range(len(collected_features)):
                # Construct inputs (collected_features[i], collected_features[j]) for varying j
                inputs = {"left": collected_features[i].unsqueeze(0).repeat(len(collected_features), 1), "right": collected_features}
                outputs = ch.nn.functional.sigmoid(meta_clf(inputs, precomputed=True)).detach().cpu().numpy()
                # Aggregate prediction as weighted sum of predicted output (which tells you whether same class or not) and corresponding ground-truth labels
                weighted_pred = outputs * signals_test_y
                # Of course, skip ith point (like, do not count it in mean)
                weighted_pred[i] = None
                test_preds.append(np.nanmean(weighted_pred))

        all_preds.append(test_preds)
    all_preds = np.array(all_preds)

    # Normalize across first dim (augmentations)
    all_preds = all_preds.mean(0)

    # Compute ROC
    auc = roc_auc_score(signals_test_y, all_preds)
    print(f"AUC: {auc}")

    # Clear memory
    ch.cuda.empty_cache()

    # Save predictions (kinda like this)
    signals_in = all_preds[signals_test_y == 1]
    signals_out = all_preds[signals_test_y == 0]

    dir_to_make = f"/p/distinf/mib_cache/signals/{args.dataset}/{args.model_arch}/0"
    if not os.path.exists(dir_to_make):
        os.makedirs(dir_to_make)
    np.save(
        f"/p/distinf/mib_cache/signals/{args.dataset}/{args.model_arch}/0/MetaAudit.npy",
        {
            "in": signals_in,
            "out": signals_out,
        },
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_arch", type=str, default="wide_resnet_28_2")
    args.add_argument("--dataset", type=str, default="cifar10")
    args.add_argument("--attack", type=str, default="MetaAudit")
    args.add_argument("--augment", action="store_true", help="Augment training data?")
    args.add_argument("--pairwise", action="store_true", help="Pairwise meta-classifier?")
    args.add_argument("--exp_seed", type=int, default=2024)
    args.add_argument("--epochs", type=int, default=50)
    args.add_argument("--target_model_index", type=int, default=0)

    args = args.parse_args()
    mp.set_start_method("spawn")
    main(args)
