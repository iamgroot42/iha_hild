import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib as mpl
import os
import argparse

mpl.rcParams["figure.dpi"] = 300

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "Times New Roman"


def main(args):
    for attack in os.listdir(args.path):
        attack_name = attack.split(".")[0]
        data = np.load(os.path.join(args.path, attack), allow_pickle=True).item()

        signals_in = data["in"]
        signals_out = data["out"]

        total_preds = np.concatenate((signals_out, signals_in))
        total_labels = [0] * len(signals_out) + [1] * len(signals_in)
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
    args.add_argument("--path", type=str, required=True)
    args.add_argument("--plotdir", type=str, default="./plots")
    args = args.parse_args()
    main(args)
