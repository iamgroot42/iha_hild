import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import argparse
import seaborn as sns
import pandas as pd
from collections import defaultdict

from mib.utils import get_signals_path

import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "Times New Roman"


ATTACKS_TO_PLOT = [
    # "LiRAOnline",
    # "LiRAOffline",
    # "LiRAOnline_aug",
    "LOSS",
    "GradNorm",
    "LiRAOnline",
    # "LiRAOnline_same_seed_ref",
    # "LiRAOnline_same_seed_ref_aug",
    # "LiRAOnline_20_ref_aug",
    # "LiRAOnline_same_seed_ref_20_ref",
    # "LiRAOnline_same_seed_ref_last5",
    # "LiRAOnline_same_seed_ref_aug_last5",
    # "UnlearnGradNorm",
    # "UnlearningAct",
    # "LiRAOnline_best5",
    # "LiRAOnline_aug_best5",
    # "LiRAOnline_last5",
    # "LiRAOnline_aug_last5",
    # "Activations",
    # "ActivationsOffline",
    "ProperTheoryRef",
]


def main(args):
    # signals_path = os.path.join(get_signals_path(), "unhinged_audit", str(args.model_index))
    signals_path = os.path.join(get_signals_path(), args.dataset, args.model_arch)

    info = defaultdict(list)
    for model_index in os.listdir(signals_path):
        inside_model_index = os.path.join(signals_path, model_index)

        for attack in os.listdir(inside_model_index):
            # Remove ".ch" from end
            attack_name = attack[:-4]
            """
            if attack_name not in ATTACKS_TO_PLOT:
                print("Skipping", attack_name, "...")
                continue
            """
            data = np.load(os.path.join(inside_model_index, attack), allow_pickle=True).item()

            signals_in = data["in"]
            signals_out = data["out"]
            total_labels = [0] * len(signals_out) + [1] * len(signals_in)

            total_preds = np.concatenate((signals_out, signals_in))
            fpr, tpr, thresholds = roc_curve(total_labels, total_preds)
            roc_auc = auc(fpr, tpr)

            tpr_at_low_fpr = lambda x: tpr[np.where(np.array(fpr) < x)[0][-1]]
            info_ = {
                "roc_auc": roc_auc,
                "tpr@0.1fpr": tpr_at_low_fpr(0.1),
                "tpr@0.01fpr": tpr_at_low_fpr(0.01),
                "tpr@0.001fpr": tpr_at_low_fpr(0.001),
                "fpr": fpr,
                "tpr": tpr,
            }
            info[attack_name].append(info_)

    """
    # First, assert we have same # of models per attack
    num_models = [len(info[attack_name]) for attack_name in info.keys()]
    if len(set(num_models)) != 1:
        raise ValueError("Different number of models per attack:", {k:len(v) for k,v in info.items()})
    print("Number of models per attack:", num_models[0])
    """

    # Plot ROC curves
    sns.set_style("whitegrid")
    sns.set_context("paper")
    sns.set_palette("tab10")

    # Aggregate results across models
    df_for_sns = []
    for attack_name, info_ in info.items():
        mean_auc = np.mean([i["roc_auc"] for i in info_])
        print(
            "%s | AUC = %0.3f ±  %0.3f | TPR@0.1FPR=%0.3f ± %0.3f | TPR@0.01FPR=%0.3f ± %0.3f | TPR@0.001FPR=%0.3f ± %0.3f"
            % (
                attack_name,
                mean_auc,
                np.std([i["roc_auc"] for i in info_]),
                np.mean([i["tpr@0.1fpr"] for i in info_]),
                np.std([i["tpr@0.1fpr"] for i in info_]),
                np.mean([i["tpr@0.01fpr"] for i in info_]),
                np.std([i["tpr@0.01fpr"] for i in info_]),
                np.mean([i["tpr@0.001fpr"] for i in info_]),
                np.std([i["tpr@0.001fpr"] for i in info_]),
            )
        )

        # TODO: Figure out proper plotting later
        """
        # Compute TPRs from each model for a given set of FPR values
        fpr_values = info_[0]["fpr"]
        tpr_values = np.zeros((len(info_), len(fpr_values)))
        for i, info_ in enumerate(info_):
            tpr_values[i] = np.array([tpr[np.where(np.array(fpr) < x)[0][-1]] for x in fpr_values])
        tpr_values = np.mean(tpr_values, axis=0)

        # Maintain DF for seaborn, since we want shaded areas for ROC curves
        for fpr, tpr in zip(fpr_values, tpr_values):
            df_for_sns.append({"FPR": fpr, "TPR": tpr, "Attack": "%s (AUC=%.3f)" % (attack_name, mean_auc)})
        """
        for fpr, tpr in zip(info_[args.which_plot]["fpr"], info_[args.which_plot]["tpr"]):
            df_for_sns.append({"FPR": fpr, "TPR": tpr, "Attack": "%s (AUC=%.3f)" % (attack_name, mean_auc)})

    exit(0)
    df_for_sns = pd.DataFrame(df_for_sns)
    sns.lineplot(data=df_for_sns, x="FPR", y="TPR", hue="Attack")

    # TODO: Increase font
    # TODO: Fix plot (current plot has shadings)
    # TODO: Change to PDF (fina)

    # Make sure plot directory exists
    if not os.path.exists(args.plotdir):
        os.makedirs(args.plotdir)

    plt.title(f"ROC Curve: {args.dataset}, {args.model_arch}")
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("TPR")
    plt.xlabel("FPR")
    plt.savefig(os.path.join(args.plotdir, f"{args.dataset}_{args.model_arch}_roc.png"))

    # Also save low TPR/FPR region
    plt.xlim([1e-5, 1e0])
    plt.ylim([1e-5, 1e0])
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(os.path.join(args.plotdir, f"{args.dataset}_{args.model_arch}_roc_lowfpr.png"))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_arch", type=str, default="wide_resnet_28_2")
    args.add_argument("--dataset", type=str, default="cifar10")
    args.add_argument("--plotdir", type=str, default="./plots")
    args.add_argument("--which_plot", type=int, default=0, help="Plot TPR-FPR curves for this specific model")
    args = args.parse_args()
    main(args)
