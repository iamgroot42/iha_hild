import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib as mpl
import os
import argparse

from mib.utils import get_signals_path

mpl.rcParams["figure.dpi"] = 300

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "Times New Roman"


def main(args):
    signals_path = os.path.join(get_signals_path(), str(args.model_index))
    for attack in os.listdir(signals_path):
        attack_name = attack.split(".")[0]
        if attack_name == "ReferenceSmooth_4_ref":
            continue
        data = np.load(os.path.join(signals_path, attack), allow_pickle=True).item()

        signals_in = data["in"]
        signals_out = data["out"]
        total_labels = [0] * len(signals_out) + [1] * len(signals_in)

        # best_auc = 0
        # best_multiplier = 1.0
        # # for multiplier in np.linspace(0.0, 1., 100):
        # for multiplier in [1.]:
        #     signals_in_ = signals_in[:, 0] - multiplier * signals_in[:, 1]
        #     signals_out_ = signals_out[:, 0] - multiplier * signals_out[:, 1]

        #     total_preds = np.concatenate((signals_out_, signals_in_))
        #     fpr, tpr, thresholds = roc_curve(total_labels, total_preds)
        #     roc_auc = auc(fpr, tpr)
        #     if roc_auc > best_auc:
        #         best_auc = roc_auc
        #         best_multiplier = multiplier
        # print("%.3f  | %.3f" % (best_multiplier, best_auc))
        # exit(0)

        total_preds = np.concatenate((signals_out, signals_in))
        fpr, tpr, thresholds = roc_curve(total_labels, total_preds)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label="%s (AUC = %0.3f)" % (attack_name, roc_auc))
        print("%s | AUC = %0.3f" % (attack_name, roc_auc))

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
    args.add_argument("--plotdir", type=str, default="./plots")
    args = args.parse_args()
    main(args)
