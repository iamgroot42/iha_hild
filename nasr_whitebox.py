"""
    White-box MI attack, as described in 'Comprehensive Privacy Analysis of Deep Learning'.
    Modified to better utilize model information. Does not use model activations.
"""
import torch as ch
import torch.nn as nn

import numpy as np
from tqdm import tqdm

from train_target import get_model_and_criterion, load_data, train_model


class MetaClassifier(nn.Module):
    def __init__(self, model, feature_dim: int = 64):
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

        # Component for output
        self.output_component = nn.Sequential(
            nn.Linear(self.num_classes_classifier, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )
        # Component for predicted class (1-hot encoding)
        self.y_component = nn.Sequential(
            nn.Linear(self.num_classes_classifier, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )
        # Component for loss
        self.loss_component = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, feature_dim)
        )
        # Component for gradients
        self.conv_grad_component = []
        for conv_shape in layers_conv:
            # For now, just flatten and throw into FC layers
            self.conv_grad_component.append(
                nn.Sequential(
                    # out, in, k, k
                    nn.Conv3d(conv_shape[0], 256, (1, conv_shape[2], conv_shape[3])),
                    nn.Flatten(),
                    nn.ReLU(),
                    nn.Linear(256 * conv_shape[1], 128),
                    nn.ReLU(),
                    nn.Linear(128, feature_dim),
                )
            )
        self.fc_grad_component = []
        for f_shape in layers_fc:
            # For now, just flatten and throw into FC layers
            self.fc_grad_component.append(
                nn.Sequential(
                    nn.Conv2d(1, 256, (1, f_shape[1])),
                    nn.Flatten(),
                    nn.ReLU(),
                    nn.Linear(256 * f_shape[0], 128),
                    nn.ReLU(),
                    nn.Linear(128, feature_dim),
                )
            )
        self.conv_grad_component = nn.ModuleList(self.conv_grad_component)
        self.fc_grad_component = nn.ModuleList(self.fc_grad_component)

        # Encoder component (combines them all)
        self.encoder = nn.Sequential(
            nn.Linear((len(layers_conv) + len(layers_fc) + 3) * feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
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
        encoder_inputs.append(self.loss_component(loss))
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
    num_train_points = 500  # 20000  # 25000
    num_test_points = 100  # 5000

    def meta_attack_per_point(index, target_model, label: int):
        meta_clf = MetaClassifier(target_model)
        # Sample num_train_points from other_indices (non-members)
        nonmember_indices = np.random.choice(
            other_indices, num_train_points, replace=False
        )
        # Sample num_train_points from train_indices except index (members)
        member_indices = np.random.choice(
            [i for i in train_index if i != index], num_train_points, replace=False
        )
        # Createe Subset datasets for these two
        nonmember_dset = ch.utils.data.Subset(all_data, nonmember_indices)
        member_dset = ch.utils.data.Subset(all_data, member_indices)
        # Createe corresponding labels (0 for nonmem, 1 for mem)
        labels = ch.tensor([0] * len(nonmember_indices) + [1] * len(member_indices))
        # Combine these two into one dataset
        combined_dset = ch.utils.data.ConcatDataset([nonmember_dset, member_dset])

        # Create meta-dset (will process and collect features)
        meta_dset = MetaDataset(
            target_model,
            combined_dset,
            labels,
            num_classes=meta_clf.num_classes_classifier,
        )

        batch_size = 64
        learning_rate = 0.001
        epochs = 20

        # Create loaders for this dset
        train_loader = ch.utils.data.DataLoader(
            meta_dset, batch_size=batch_size, shuffle=True
        )
        criterion = nn.BCEWithLogitsLoss()

        # Train meta-classifier
        meta_clf = train_model(
            meta_clf.cuda(),
            criterion,
            train_loader,
            test_loader=None,
            learning_rate=learning_rate,
            epochs=epochs,
        )

        # Collect feature for datapoint to be tested
        features = meta_dset._collect_features(
            all_data[index][0].unsqueeze(0).cuda(),
            ch.tensor([label]).cuda(),
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
        signals_in.append(meta_attack_per_point(index, model, 1))
    for index in other_indices:
        signals_out.append(meta_attack_per_point(index, model, 0))

    # Prepare data for saving
    signals = np.concatenate((signals_out, signals_in))
    labels = np.concatenate((np.zeros(len(signals) // 2), np.ones(len(signals) // 2)))

    # Save both these arrays
    np.save("signals_pin.npy", {"signals": signals, "labels": labels})


if __name__ == "__main__":
    main()
