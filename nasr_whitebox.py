"""
    White-box MI attack, as described in 'Comprehensive Privacy Analysis of Deep Learning'.
    Modified to better utilize model information. Does not use model activations.
"""
import torch as ch
import torch.nn as nn

import numpy as np
from tqdm import tqdm

from mib.train import get_model_and_criterion, load_data, train_model


class MetaClassifier(nn.Module):
    def __init__(self, model, feature_dim: int):
        """
        Use model skeleton to decide architecture of meta-classifier.
        """
        super().__init__()

        # Personally I think a non-linearity in the loss and label encodings are overkill
        layers_conv, layers_fc = [], []
        for n, p in model.named_parameters():
            # If Conv layer, take note of shape
            if "conv" in n and ".weight" in n:
                layers_conv.append(p.shape)
            # If FC layer, take note of shape
            elif "linear" in n and ".weight" in n:
                layers_fc.append(p.shape)

        # Inger num_classes_classifier from model
        self.num_classes_classifier = layers_fc[-1][0]

        # Feature dimension for small inputs
        self.small_feature_dim = 4

        num_features_pre_final = 0

        # Component for output
        self.output_component = nn.Sequential(
            nn.Linear(self.num_classes_classifier, self.small_feature_dim), nn.ReLU()
        )
        num_features_pre_final += self.small_feature_dim

        # Component for predicted class (1-hot encoding)
        self.y_component = nn.Sequential(
            nn.Linear(self.num_classes_classifier, self.small_feature_dim), nn.ReLU()
        )
        num_features_pre_final += self.small_feature_dim

        # Loss-component makes no sense- remove
        """
        # Component for loss
        self.loss_component = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
        num_features_pre_final += feature_dim
        """
        num_features_pre_final += 1

        # Component for gradients
        self.conv_grad_component = []
        for conv_shape in layers_conv:
            # For now, just flatten and throw into FC layers
            self.conv_grad_component.append(
                nn.Sequential(
                    # out, in, k, k
                    nn.Conv3d(conv_shape[0], 32, (1, conv_shape[2], conv_shape[3])),
                    nn.Flatten(),
                    nn.ReLU(),
                    nn.Linear(32 * conv_shape[1], 32),
                    nn.ReLU(),
                    nn.Linear(32, feature_dim),
                )
            )
            num_features_pre_final += feature_dim
        self.fc_grad_component = []
        for f_shape in layers_fc:
            # For now, just flatten and throw into FC layers
            self.fc_grad_component.append(
                nn.Sequential(
                    nn.Conv2d(1, 32, (1, f_shape[1])),
                    nn.Flatten(),
                    nn.ReLU(),
                    nn.Linear(32 * f_shape[0], 32),
                    nn.ReLU(),
                    nn.Linear(32, feature_dim),
                )
            )
            num_features_pre_final += feature_dim

        self.conv_grad_component = nn.ModuleList(self.conv_grad_component)
        self.fc_grad_component = nn.ModuleList(self.fc_grad_component)

        # Encoder component (combines them all)
        self.encoder = nn.Sequential(
            nn.Linear(num_features_pre_final, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, data):
        """
        Forward pass of meta-classifier.
        """
        # Unpack data
        logits, loss, y, gradients_conv, gradients_fc = data

        encoder_inputs = []
        encoder_inputs.append(self.output_component(logits))
        encoder_inputs.append(self.y_component(y))
        # encoder_inputs.append(self.loss_component(loss))
        encoder_inputs.append(loss)
        for i, grad in enumerate(gradients_conv):
            encoder_inputs.append(self.conv_grad_component[i](grad))
        for i, grad in enumerate(gradients_fc):
            encoder_inputs.append(self.fc_grad_component[i](grad.unsqueeze(1)))
        # Concatenate all components
        a = ch.cat(encoder_inputs, 1)
        logits = self.encoder(a)
        return logits


class MetaDataset(ch.utils.data.Dataset):
    def __init__(self, model, dset, labels, num_classes: int):
        """
        Dataset for meta-classifier.
        """
        self.model = model
        self.features = []
        self.labels = labels
        self.num_classes = num_classes
        # Collect self.features array with all collected features
        for x, y in tqdm(dset, desc="Collecting features for meta-classifier"):
            self.features.append(
                self._collect_features(x.unsqueeze(0).cuda(), ch.tensor([y]).cuda())
            )

    def _collect_features(self, x, y):
        """
        Collect features for meta-classifier.
        """
        # Get model output
        logits = self.model(x)
        # Get loss
        loss = nn.CrossEntropyLoss()(logits, y.long())
        loss.backward()
        # Get gradients
        gradients_conv, gradients_fc = [], []
        for n, p in self.model.named_parameters():
            # If Conv layer, take note of shape
            if "conv" in n and ".weight" in n:
                gradients_conv.append(p.grad.detach().cpu())
            # If FC layer, take note of shape
            elif "linear" in n and ".weight" in n:
                gradients_fc.append(p.grad.detach().cpu())
        self.model.zero_grad()

        logits = logits.detach().cpu().squeeze(0)
        loss = loss.detach().cpu().unsqueeze(0)
        y = nn.functional.one_hot(y.detach().cpu(), self.num_classes).float().squeeze(0)

        return logits, loss, y, gradients_conv, gradients_fc

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def main():
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
    feature_dim = 16
    num_classes = 10
    num_samples_test = 50

    # Create Subset datasets for members
    member_dset = ch.utils.data.Subset(all_data, train_index)
    # and non- members
    nonmember_indices = np.random.choice(other_indices, num_train_points, replace=False)
    nonmember_dset = ch.utils.data.Subset(all_data, nonmember_indices)

    # Keep track of all indeces as well
    indices_together = np.concatenate((train_index, nonmember_indices))

    # Create corresponding labels (1 for mem, 0 for nonmem).
    labels = ch.tensor([1] * len(train_index) + [0] * len(nonmember_indices))
    # Combine these two into one dataset
    combined_dset = ch.utils.data.ConcatDataset([member_dset, nonmember_dset])
    # Only need to collect features once
    meta_dset = MetaDataset(
        model,
        combined_dset,
        labels,
        num_classes=num_classes,
    )

    def meta_attack_per_point(index, target_model):
        # Prepare to train meta-classifier
        meta_clf = MetaClassifier(target_model, feature_dim=feature_dim).cuda()
        # Compile for faster training
        # meta_clf = ch.compile(meta_clf)

        # Remove index being tested (find position first)
        position_to_remove = np.where(np.array(indices_together) == index)[0][0]
        all_indices = np.arange(len(meta_dset))
        all_indices = np.delete(all_indices, position_to_remove)
        meta_dset_use = ch.utils.data.Subset(meta_dset, all_indices)

        # Create mask of length labels where index position
        batch_size = 512
        learning_rate = 0.01
        epochs = 30

        # Create loaders for this dset
        train_loader = ch.utils.data.DataLoader(
            meta_dset_use,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Train meta-classifier
        meta_clf = train_model(
            meta_clf,
            criterion,
            train_loader,
            test_loader=None,
            learning_rate=learning_rate,
            epochs=epochs,
        )

        # Collect feature for datapoint to be tested
        features = meta_dset._collect_features(
            all_data[index][0].unsqueeze(0).cuda(),
            ch.tensor([all_data[index][1]]).cuda(),
        )
        # Unsqueeze all of them
        features_send = []
        for f in features:
            if type(f) == list:
                features_send.append([x.unsqueeze(0) for x in f])
            else:
                features_send.append(f.unsqueeze(0))
        # Get meta-classifier's prediction
        with ch.no_grad():
            meta_clf.eval()
            output = meta_clf(features_send).detach()
            output = ch.sigmoid(output).item()
        return output

    signals_in, signals_out = [], []
    for index in train_index:
        signals_in.append(meta_attack_per_point(index, model))
        if len(signals_in) >= num_samples_test:
            break
    for index in nonmember_indices:
        signals_out.append(meta_attack_per_point(index, model))
        if len(signals_out) >= num_samples_test:
            break

    print(signals_in)
    print(signals_out)

    # Prepare data for saving
    signals = np.concatenate((signals_out, signals_in))
    labels = np.concatenate((np.zeros(len(signals) // 2), np.ones(len(signals) // 2)))

    # Save both these arrays
    np.save("signals_nasr.npy", {"signals": signals, "labels": labels})


if __name__ == "__main__":
    main()
