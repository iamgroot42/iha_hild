import torch as ch
import numpy as np
import os
import argparse
import copy

from mib.models.utils import get_model
from mib.utils import get_models_path
from mib.dataset.utils import get_dataset
from mib.train import get_loader, train_model


def main(save_dir, args):
    pkeep = 0.9
    ds = get_dataset("purchase100")()
    device = "cuda"
    n_data_samples = 32

    if args.loo_point is not None:
        save_dir_use = os.path.join(save_dir, f"loo", str(args.loo_point))
    else:
        save_dir_use = os.path.join(save_dir, f"all")

    assert args.data_sample_idx < n_data_samples, f"data_sample_idx must be < {n_data_samples}"

    # Get data
    train_index, test_index, train_data, test_data = ds.make_splits(
        seed=args.data_seed, num_experiments=n_data_samples, pkeep=pkeep, exp_id=args.data_sample_idx
    )

    def get_save_dict(z):
        # z is a DP wrapped (on top of compile-wrapped) dict
        # we want state_dict of the original model
        # return z._orig_mod.state_dict()
        return z.module.state_dict()

    # LOO setting
    if args.loo_point is not None:
        train_index = np.delete(train_index, args.loo_point)

    # Make sure folder directory exists
    os.makedirs(save_dir_use, exist_ok=True)

    for i in range(args.trials):
        # Skip model training if path exists
        if any(
            [os.path.exists(f"{save_dir_use}/{i}_{j}.pt") for j in range(args.pick_n)]
        ):
            print("Skipping!")
            continue

        model, criterion, hparams = get_model(args.model_arch, n_classes=ds.num_classes)
        model.to(device)

        # To be able to compare Hessian across LOO setting, we ideally want same starting randomness of weights
        # To minimize divergence into functionally-equivalent NNs (permutation invariance)
        if args.loo_point is not None:
            # Use same initialization as model with index same_init
            model_to_load = f"{save_dir}/all/{i}_0.pt"
            if not os.path.exists(model_to_load):
                raise ValueError(f"Model {model_to_load} not found, should be trained and saved before moving to LOO setting")
            model_init = ch.load(f"{save_dir}/all/{i}_0.pt")["model_init"]
            model.load_state_dict(model_init)
            model.to(device)
        else:
            # Random initialization
            model_init = copy.deepcopy(model.state_dict())

        batch_size = hparams["batch_size"]
        learning_rate = hparams["learning_rate"]
        epochs = hparams["epochs"]

        train_loader = get_loader(train_data, train_index, batch_size)
        test_loader = get_loader(test_data, test_index, batch_size)

        # Utilize DDP
        model = ch.nn.parallel.DataParallel(model)

        # Train model
        models, accs, losss = train_model(
            model,
            criterion,
            train_loader,
            test_loader,
            learning_rate,
            epochs,
            get_final_model=False,
            use_scheduler=False,
            opt_momentum=0,
            weight_decay=0,
            pick_mode="last_n",
            pick_n=args.pick_n
        )

        # Save model checkpoints
        for j, (model, acc, loss) in enumerate(zip(models, accs, losss)):
            # Save models
            ch.save(
                {
                    "model_init": model_init,
                    "model": get_save_dict(model),
                    "train_index": train_index,
                    "test_index": test_index,
                    "loss": loss,
                    "acc": acc,
                },
            f"{save_dir_use}/{i}_{j}.pt")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_arch", type=str, default="mlp4")
    args.add_argument("--dataset", type=str, default="purchase100")
    args.add_argument("--data_seed", type=int, default=2024)
    args.add_argument("--data_sample_idx", type=int, default=0, help="Model idx (data sample). Must be < 32")
    args.add_argument("--trials", type=int, default=10, help="Number of models to train (out of num_models)")
    args.add_argument("--loo_point", type=int, default=None, help="Index for LOO-setting training")
    args.add_argument(
        "--pick_n", type=int, default=20, help="Of all checkpoints, keep n."
    )
    args = args.parse_args()

    save_dir = get_models_path()
    main(
        os.path.join(
            save_dir, "theory_exps", args.dataset, args.model_arch, str(args.data_sample_idx)
        ),
        args,
    )
