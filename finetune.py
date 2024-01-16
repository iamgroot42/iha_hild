"""
    Main idea is to think of the original model (when record is absent) as a perfect "unlearned" version of a model that does indeed contain the record.
    This can be achieved by:
    1. Explicitly fine-tuning on the target record (with some other data) for a few batches, and see how different this is from the starting model.
    2. Explicitly perform model unlearning.
"""
import torch as ch
import copy
import numpy as np
from tqdm import tqdm
import torch.nn as nn

from bbeval.config import ExperimentConfig

from mi_benchmark.train import get_model_and_criterion, get_data, train_model


# Create a CustomSampler that always includes some point X in batch, and samples remaining points from other_data_source
class SpecificPointIncludedLoader:
    def __init__(self, given_loader, interest_point, num_batches: int):
        self.given_loader = given_loader
        self.interest_point = interest_point
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            x, y = next(iter(self.given_loader))
            # x, y are torch tensors
            # append x, y into these tensors
            x = ch.cat([x, self.interest_point[0].unsqueeze(0)])
            y = ch.cat([y, ch.tensor([self.interest_point[1]])])
            batch = (x, y)
            yield batch

    def __len__(self):
        return self.num_batches


def finetune(
    model,
    interest_point,
    other_data_source,
    num_times: int,
    learning_rate: float,
    batch_size: int = 256,
):
    """
    Create N (where model was trained for N epochs) batches of data where 1 point is of interest,
    and others are sampled each time from 'other_data_source'. Send this loader over to training,
    effectively fine-tuning the model on the target record (and other misc records). Keep track of
    change in model performance on this point before and after fine-tuning.
    """
    # Create loader out of given data source
    given_loader = ch.utils.data.DataLoader(
        other_data_source, batch_size=batch_size - 1, shuffle=True
    )
    # Create custom loader
    loader = SpecificPointIncludedLoader(given_loader, interest_point, num_times)
    criterion = nn.CrossEntropyLoss()
    # Make copy of model before sending it over
    model_ = copy.deepcopy(model.cpu())

    def measurement(a, b):
        # Get model parameters
        params = []
        for p in model_.parameters():
            params.extend(list(p.detach().cpu().numpy().flatten()))
        params = np.array(params)

        # Get model prediction (will use to compare later)
        pred_softmax = ch.nn.functional.softmax(
            model_(a).detach().cpu(), dim=1
        ).numpy()[0]
        # Measure loss
        loss_measure = criterion(model_(a), b).detach().cpu().item()

        # Compute gradient norm with given (a, b) datapoint
        model_.zero_grad()
        pred = model_(a)
        loss = criterion(pred, b)
        loss.backward()
        grads = []
        for p in model_.parameters():
            grads.extend(list(p.grad.detach().cpu().numpy().flatten()))
        model_.zero_grad()
        grads = np.array(grads)

        return {
            "pred": pred_softmax,
            "loss": loss_measure,
            "grads": grads,
            "params": params,
        }

    # Take note of loss for interest_point before fine-tuning
    signals_before = measurement(
        interest_point[0].unsqueeze(0), ch.tensor([interest_point[1]])
    )

    model_.cuda()
    model_, _, _ = train_model(
        model_,
        criterion,
        loader,
        None,
        learning_rate=learning_rate,
        epochs=1,
        verbose=False,
        loss_multiplier=-1,
    )
    # Do sth with new model
    signals_after = measurement(
        interest_point[0].unsqueeze(0), ch.tensor([interest_point[1]])
    )

    return {
        "before": signals_before,
        "after": signals_after,
    }

    # Compute similarity between both pred probs
    # Use l-2 norm
    # return np.linalg.norm(score_before - score_after)
    # Use cross-entropy
    # return -np.sum(score_before * np.log(score_after + 1e-8))

    # Entropy before, after, and drop
    # ent_before = -np.sum(score_before * np.log(score_before + 1e-8))
    # ent_after = -np.sum(score_after * np.log(score_after + 1e-8))
    # return ent_before, ent_after, ent_before - ent_after

    # Compute grad norm before, grad norm after, and norm of grad diffs
    # before = np.linalg.norm(score_before)
    # after = np.linalg.norm(score_after)
    # diff = np.linalg.norm(score_after - score_before)
    # return before, after, diff


def main():
    currdir = "."
    currdir = "/u/as9rw/work/auditing_mi/"
    config = ExperimentConfig.load(currdir + "smimifgsm.json", drop_extra_fields=False)

    # Load target model
    model_dict = ch.load("./models/0.pt")
    # Extract member information and model
    model_weights = model_dict["model"]
    model, _ = get_model_and_criterion("cifar10", device="cpu")
    model.load_state_dict(model_weights, strict=False)
    # Shift to CUDA
    model = model.cuda()
    # Make sure it's on eval model
    model.eval()

    train_index = model_dict["train_index"]

    # Get data
    train_data, _ = get_data(just_want_data=True)
    # Get indices out of range(len(train_data)) that are not train_index or test_index
    other_indices_train = np.array(
        [i for i in range(len(train_data)) if i not in train_index]
    )

    # CIFAR
    num_train_points = 10000
    num_samples_test = 500
    learning_rate = 0.001
    epochs = 5  # 20
    batch_size = 256

    # Create Subset datasets for members
    member_dset = ch.utils.data.Subset(train_data, train_index)
    # and non- members
    nonmember_indices = np.random.choice(
        other_indices_train, num_train_points, replace=False
    )
    # Break nonmember_indices here into 2 - one for sprinkling in FT data, other for actual non-members
    nonmember_indices_ft = nonmember_indices[: num_train_points // 2]
    nonmember_indices_test = nonmember_indices[num_train_points // 2 :]

    nonmember_dset_ft = ch.utils.data.Subset(train_data, nonmember_indices_ft)
    nonmember_dset = ch.utils.data.Subset(train_data, nonmember_indices_test)

    signals_in, signals_out = [], []
    for mem in tqdm(
        member_dset, total=num_samples_test, desc="Collecting member signals"
    ):
        signals_in.append(
            finetune(
                model,
                mem,
                nonmember_dset_ft,
                epochs,
                learning_rate,
                batch_size=batch_size,
            )
        )
        if len(signals_in) >= num_samples_test:
            break
    for nonmem in tqdm(
        nonmember_dset,
        total=num_samples_test,
        desc="Collecting non-member signals",
    ):
        signals_out.append(
            finetune(
                model,
                nonmem,
                nonmember_dset_ft,
                epochs,
                learning_rate,
                batch_size=batch_size,
            )
        )
        if len(signals_out) >= num_samples_test:
            break

    signals_in = np.array(signals_in)
    signals_out = np.array(signals_out)
    np.save(f"signals/unlearn/{epochs}e_{batch_size}b.npy", {"in": signals_in, "out": signals_out})


if __name__ == "__main__":
    main()
