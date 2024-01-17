import torch as ch
import numpy as np
from tqdm import tqdm
import copy

from mib.train import (
    get_dataset_subset,
    get_model_and_criterion,
    load_data,
    train_model,
    get_loaders,
)

from distribution_inference.attacks.whitebox.permutation.permutation import PINAttack
from distribution_inference.config.core import (
    WhiteBoxAttackConfig,
    PermutationAttackConfig,
)
from utils import get_weight_layers, wrap_into_loader

# from distribution_inference.attacks.whitebox.utils import (
#     get_weight_layers,
#     wrap_into_loader,
# )


def sample_points_excluding(num_points_total: int, index_exclude: int):
    pool = np.arange(num_points_total)
    # Remove index_exclude from pool
    pool = np.array([i for i in pool if i != index_exclude], dtype=np.int32)
    train_index = np.random.choice(pool, num_train_points, replace=False)
    # Remove index_exclude from train_index (if present)
    train_index = np.array(
        [i for i in train_index if i != index_exclude], dtype=np.int32
    )
    test_index = np.random.choice(
        [i for i in pool if i not in train_index], num_test_points, replace=False
    )
    # Repeat same exclusion for index_exclude
    test_index = np.array([i for i in test_index if i != index_exclude], dtype=np.int32)
    return train_index, test_index


if __name__ == "__main__":
    # Load target model
    model_dict = ch.load("target_model/1.pt")
    # Extract member information and model
    model_weights = model_dict["model"]
    model, criterion = get_model_and_criterion("cifar10", device="cpu")
#     model.load_state_dict(model_weights, strict=False)

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

    n_points_test = 10
    n_shadow_models_each = 2

    # CIFAR
    num_train_points = 20000 #25000
    num_test_points = 100 #5000
    batch_size = 64
    learning_rate = 0.001
    epochs = 20

    def pin_attack_per_point(index, target_model, label: int):
        models_without = []
        models_with = []

        # Train shadow models
        for i in tqdm(range(n_shadow_models_each), desc="Training shadow models"):
            ### Models without target member

            # Sample shadow dataset
            shadow_points_train, shadow_points_test = sample_points_excluding(
                len(all_data), index
            )
            # Create loader for shadow dataset
            train_loader, test_loader = get_loaders(
                all_data, shadow_points_train, shadow_points_test, batch_size
            )
            # Train shadow model
            model_ = train_model(
                copy.deepcopy(model).cuda(), criterion, train_loader, test_loader, learning_rate, epochs
            )
            models_without.append(model_.cpu())

            ## Models with target member
            shadow_points_train = np.append(shadow_points_train, [index])
            # Create loader for shadow dataset
            train_loader, test_loader = get_loaders(
                all_data, shadow_points_train, shadow_points_test, batch_size
            )
            # Train shadow model
            model_ = train_model(
                copy.deepcopy(model).cuda(), criterion, train_loader, test_loader, learning_rate, epochs
            )
            models_with.append(model_.cpu())

        # Train meta-classifier (PIN-based) that uses model parameters to predict whether given point
        # was a member or not
        # Then, use this meta-classifier to predict membership for target model
        perm_config = PermutationAttackConfig(focus="all")
        wb_config = WhiteBoxAttackConfig(
            attack="permutation_invariant",
            epochs=100,
            batch_size=n_shadow_models_each * 2,
            dropout=0,
            permutation_config=perm_config,
        )

        features_in, features_out = [], []
        for model_ in models_without:
            dims, fo = get_weight_layers(model_, wb_config)
            features_out.append(fo)
        for model_ in models_with:
            dims, fi = get_weight_layers(model_, wb_config)
            features_in.append(fi)
        # Wrap into one loader
        train_loader = wrap_into_loader(
            [features_out, features_in],
            batch_size=n_shadow_models_each * 2,
            shuffle=False,
        )

        # Test loader only comprises of target model
        test_loader = wrap_into_loader(
            [target_model], batch_size=1, labels_list=[float(label)]
        )

        attack = PINAttack(dims, wb_config)
        attack.execute_attack(train_loader)
        with ch.no_grad():
            attack.model.eval()
            # Get P(member)
            test_model = next(iter(test_loader))[0]
            test_model = [a.cuda() for a in test_model]
            output = attack.model(test_model).item()
            return output

    signals_in, signals_out = [], []
    for index in train_index:
        signals_in.append(pin_attack_per_point(index, model, 1))
    for index in other_indices:
        signals_out.append(pin_attack_per_point(index, model, 0))

    # Prepare data for saving
    signals = np.concatenate((signals_out, signals_in))
    labels = np.concatenate((np.zeros(len(signals) // 2), np.ones(len(signals) // 2)))

    # Save both these arrays
    np.save("signals_pin.npy", {"signals": signals, "labels": labels})
