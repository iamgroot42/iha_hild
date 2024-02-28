"""
    Use knowledge of membership from X% of actual training data to train membership meta-classifier.
    Potentially unrealistic threat model as an adversary, but plausible for auditing.
"""
import os
import torch as ch
import numpy as np
import argparse
from tqdm import tqdm


from mib.attack import member_nonmember_loaders
from mib.models.utils import get_model
from mib.dataset.utils import get_dataset
from mib.attacks.utils import get_attack
from mib.utils import get_signals_path, get_models_path

from mib.attacks.meta_audit import train_meta_clf, MetaModePlain, MetaModelCNN


def get_signals(model, x, y):
    LAYER_READOUT = 0
    acts = model(x.cuda(), layer_readout=LAYER_READOUT).detach().cpu()
    # Also get loss
    loss = ch.nn.CrossEntropyLoss()(model(x.cuda()), y.cuda()).detach().cpu()
    # Concatenate and return
    return acts, loss.unsqueeze(0)


def pointwise_audit(x_train, y_train, x_test_in, x_test_out):
    # Create meta-classifier
    num_feats = x_train[0][0].shape[0]
    meta_clf = MetaModelCNN(input_dim=num_feats)
    meta_clf = train_meta_clf(meta_clf, x_train, y_train, batch_size=128, val_points=25)
    meta_clf.eval()
    meta_clf.cuda()
    with ch.no_grad():
        # Scores for 'in' point
        inputs = [ch.tensor(z).cuda() for z in x_test_in]
        score_in = meta_clf(*inputs).detach().cpu().numpy()
        # Scores for some unseen 'out' points
        inputs = [ch.tensor(z).cuda() for z in x_test_out]
        scores_out = meta_clf(*inputs).detach().cpu().numpy()
    return score_in, scores_out


def main(args):
    model_dir = os.path.join(get_models_path(), args.model_arch)

    # Load target model
    target_model, _, _ = get_model(args.model_arch, 10)
    model_dict = ch.load(os.path.join(model_dir, f"{args.target_model_index}.pt"))
    target_model.load_state_dict(model_dict["model"], strict=False)
    target_model.eval()

    # Pick records (out of all train) to test
    train_index = model_dict["train_index"]

    # TODO: Get these stats from a dataset class
    ds = get_dataset(args.dataset)(augment=False)
    # CIFAR
    num_nontrain_pool = 25000 #10000

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
        want_all_member_nonmember=True
    )

    # For reference-based attacks, train out models
    # attacker = get_attack(args.attack)(target_model)
    target_model.cuda()

    # Compute signals for member data
    signals_in, signals_out = [], []
    for i, (x, y) in tqdm(enumerate(member_loader), total=len(member_loader)):
        signal = get_signals(target_model, x, y)
        signals_in.append(signal)
    signals_in = zip(*signals_in)
    signals_in = [np.concatenate(x, 0) for x in signals_in]

    # Compute signals for non-member data
    for i, (x, y) in tqdm(enumerate(nonmember_loader), total=len(nonmember_loader)):
        signal = get_signals(target_model, x, y)
        signals_out.append(signal)
    signals_out = zip(*signals_out)
    signals_out = [np.concatenate(x, 0) for x in signals_out]

    # Flatten out the signals (take max over last 2 dims)
    # signals_in  = signals_in.reshape(signals_in.shape[0], -1)
    # signals_out = signals_out.reshape(signals_out.shape[0], -1)
    # signals_in = np.max(np.max(signals_in, 2), 2)
    # signals_out = np.max(np.max(signals_out, 2), 2)

    min_of_both = min(len(signals_in[0]), len(signals_out[0]))
    NUM_HOLDOUT_TEST = 100
    random_idx = np.random.permutation(min_of_both)
    train_idx = random_idx[NUM_HOLDOUT_TEST:]
    test_idx = random_idx[:NUM_HOLDOUT_TEST]
    signals_train_x = [np.concatenate([a[train_idx], b[train_idx]], 0) for a, b in zip(signals_in, signals_out)]
    signals_test_x = [z[test_idx] for z in signals_out]
    signals_train_y = np.concatenate(
        [np.ones(len(train_idx)), np.zeros(len(train_idx))]
    )

    # Make sure they're long-type
    num_tryout = args.num_points
    scores_in_all, scores_out_all = [], []
    for i in range(num_tryout):
        # Set ith point to be test, and rest to be train
        test_idx = [i]
        train_idx = [j for j in range(len(signals_train_x[0])) if j != i]
        train_x = [z[train_idx] for z in signals_train_x]
        train_y = signals_train_y[train_idx]
        test_x = [z[i] for z in signals_train_x]
        score_in, scores_out = pointwise_audit(train_x, train_y, test_x, signals_test_x)
        scores_in_all.append(score_in)
        scores_out_all.append(np.mean(scores_out))
        print(score_in, np.mean(scores_out))

    """
    signals_dir = get_signals_path()
    save_dir = os.path.join(signals_dir, "unhinged_audit", str(args.target_model_index))

    # Make sure save_dir exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    attack_name = args.attack

    # signals_in are not predictins corresponding to signals_test_y == 1
    # signals_out are not predictins corresponding to signals_test_y == 0
    signals_in  = scores[signals_test_y == 1]
    signals_out = scores[signals_test_y == 0]

    np.save(
        f"{save_dir}/{attack_name}.npy",
        {
            "in": signals_in,
            "out": signals_out,
        },
    )
    """


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_arch", type=str, default="wide_resnet_28_2")
    args.add_argument("--dataset", type=str, default="cifar10")
    args.add_argument("--attack", type=str, default="MetaAudit")
    args.add_argument("--exp_seed", type=int, default=2024)
    args.add_argument("--target_model_index", type=int, default=0)
    args.add_argument(
        "--num_points",
        type=int,
        default=500,
        help="Number of samples (in and out each) for mete-classifier testing",
    )
    args = args.parse_args()
    main(args)
