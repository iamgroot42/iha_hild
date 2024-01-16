"""
    Evaluating MI attacks using privacy_meter. Code heavily based on https://github.com/privacytrustlab/ml_privacy_meter/blob/master/advanced/white_box_attack.ipynb
"""
import numpy as np

import torch.nn.functional as F
from torch import nn
import torchvision
from ast import List
import time
import numpy as np
import torch
import torchvision
from torchvision import transforms

from privacy_meter.audit import Audit, MetricEnum
from privacy_meter.metric import PopulationMetric
from privacy_meter.information_source_signal import (
    ModelGradientNorm,
    ModelGradient,
    ModelLoss,
)
from privacy_meter.hypothesis_test import linear_itp_threshold_func
from privacy_meter.audit_report import ROCCurveReport
from privacy_meter.constants import InferenceGame
from privacy_meter.dataset import Dataset
from privacy_meter.information_source import InformationSource
from privacy_meter.model import PytorchModelTensor


# for training the target model
num_train_points = 5000
num_test_points = 5000
num_population_points = 10000
batch_size = 64
learning_rate = 0.001
epochs = 50


def population_attack(target_info_source, reference_info_source):
    # Attack based on population metric
    metric = PopulationMetric(
        target_info_source=target_info_source,
        reference_info_source=reference_info_source,
        signals=[ModelLoss()],
        hypothesis_test_func=linear_itp_threshold_func,
    )
    audit_obj = Audit(
        metrics=metric,
        inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
        target_info_sources=target_info_source,
        reference_info_sources=reference_info_source,
    )
    audit_obj.prepare()
    results = audit_obj.run()[0]
    return results


def grad_norm_attack(target_info_source, reference_info_source):
    # Attack based on gradient norm
    metric = PopulationMetric(
        target_info_source=target_info_source,
        reference_info_source=reference_info_source,
        signals=[ModelGradientNorm()],
        hypothesis_test_func=linear_itp_threshold_func,
    )
    audit_obj = Audit(
        metrics=metric,
        inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
        target_info_sources=target_info_source,
        reference_info_sources=reference_info_source,
    )
    audit_obj.prepare()
    results = audit_obj.run()[0]
    return results


def get_dataset_subset(dataset: torchvision.datasets, index: List(int)):
    """Get a subset of the dataset.

    Args:
        dataset (torchvision.datasets): Whole dataset.
        index (list): List of index.
    """
    assert max(index) < len(dataset) and min(index) >= 0, "Index out of range"
    data = (
        torch.from_numpy(dataset.data[index]).float().permute(0, 3, 1, 2) / 255
    )  # channel first
    targets = list(np.array(dataset.targets)[index])
    targets = torch.tensor(targets, dtype=torch.long)
    return data, targets


def main():
    dataset = "CIFAR10"
    device = "cuda"

    # Get model
    model, criterion = get_model_and_criterion(dataset, device=device)

    # Get data
    all_data, train_index, test_index, population_index = load_data("CIFAR10")

    # Get loaders
    train_loader, test_loader = get_loaders(all_data, train_index, test_index)

    # Train model
    model = train_model(model, criterion, train_loader, test_loader)

    # Create data splits
    train_data, train_targets = get_dataset_subset(all_data, train_index)
    test_data, test_targets = get_dataset_subset(all_data, test_index)
    audit_data, audit_targets = get_dataset_subset(all_data, population_index)

    target_dataset = Dataset(
        data_dict={
            "train": {"x": train_data, "y": train_targets},
            "test": {"x": test_data, "y": test_targets},
        },
        default_input="x",
        default_output="y",
    )

    audit_dataset = Dataset(
        data_dict={"train": {"x": audit_data, "y": audit_targets}},
        default_input="x",
        default_output="y",
    )

    # Wrap model
    target_model = PytorchModelTensor(
    model_obj=model, loss_fn=criterion, device=device, batch_size=10
    )

    target_info_source = InformationSource(
        models=[target_model], datasets=[target_dataset]
    )

    reference_info_source = InformationSource(
        models=[target_model], datasets=[audit_dataset]
    )

    # Over all attacks
    attacks = {
        "population": population_attack,
        "gradient_norm": grad_norm_attack,
    }
    for name, attack in attacks.items():
        result = attack(target_info_source, reference_info_source)
        ROCCurveReport.generate_report(
            metric_result=result[0],
            inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
            show=True,
        )


# Compare results
# from privacy_meter import audit_report

# This instruction won't be needed once the tool is on pip
# audit_report.REPORT_FILES_DIR = "../privacy_meter/report_files"
