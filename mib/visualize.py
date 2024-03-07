import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
import os
import argparse

from mib.utils import get_signals_path

mpl.rcParams["figure.dpi"] = 300

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "Times New Roman"


ATTACKS_TO_PLOT = [
    # "LiRAOnline",
    "LiRAOnline_aug",
    "LOSS",
    "LiRAOnline_same_seed_ref",
    "LiRAOnline_same_seed_ref_aug",
    # "LiRAOnline_20_ref_aug",
    # "LiRAOnline_same_seed_ref_20_ref",
    # "LiRAOnline_same_seed_ref_last5",
    # "LiRAOnline_same_seed_ref_aug_last5",
    # "UnlearnGradNorm",
    "UnlearningAct",
    # "LiRAOnline_best5",
    # "LiRAOnline_aug_best5",
    "LiRAOnline_last5",
    # "LiRAOnline_aug_last5",
    # "Activations",
    "ActivationsOffline",
    "TheoryRef"
]


def main(args):
    # signals_path = os.path.join(get_signals_path(), "unhinged_audit", str(args.model_index))
    signals_path = os.path.join(get_signals_path(), args.dataset, str(args.model_index))

    info = {}
    for attack in os.listdir(signals_path):
        attack_name = attack.split(".")[0]
        """
        if attack_name not in ATTACKS_TO_PLOT:
            print("Skipping", attack_name, "...")
            continue
        """
        data = np.load(os.path.join(signals_path, attack), allow_pickle=True).item()

        signals_in = data["in"]
        signals_out = data["out"]
        total_labels = [0] * len(signals_out) + [1] * len(signals_in)

        total_preds = np.concatenate((signals_out, signals_in))
        fpr, tpr, thresholds = roc_curve(total_labels, total_preds)

        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label="%s (AUC = %0.3f)" % (attack_name, roc_auc))

        tpr_at_low_fpr = lambda x: tpr[np.where(np.array(fpr) < x)[0][-1]]
        info_ = {
            "roc_auc": roc_auc,
            "tpr@0.1fpr": tpr_at_low_fpr(0.1),
            "tpr@0.01fpr": tpr_at_low_fpr(0.01)
        }
        info[attack_name] = info_

        # print(
        #     "%s | AUC = %0.3f | TPR@0.1FPR=%0.3f | TPR@0.01FPR=%0.3f"
        #     % (attack_name, roc_auc, info_["tpr@0.1fpr"], info_["tpr@0.01fpr"])
        # )

    # Make sure plot directory exists
    if not os.path.exists(args.plotdir):
        os.makedirs(args.plotdir)

    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("TPR")
    plt.xlabel("FPR")
    plt.savefig(os.path.join(args.plotdir, "roc.png"))

    # Also save low TPR/FPR region
    plt.xlim([1e-5, 1e0])
    plt.ylim([1e-5, 1e0])
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(os.path.join(args.plotdir, "roc_lowfpr.png"))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_index", type=int, default=0)
    args.add_argument("--dataset", type=str, default="cifar10")
    args.add_argument("--plotdir", type=str, default="./plots")
    args = args.parse_args()
    main(args)
