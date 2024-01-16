import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "Times New Roman"


def main():
    z = np.load("./signals/unlearn/5e_256b.npy", allow_pickle=True).item()

    signals_in = z["in"]
    signals_out = z["out"]

    def entropy(z):
        return -np.sum(z * np.log(z + 1e-30))

    attacks = {
        "LOSS": 
        lambda x: -x["before"]["loss"],
        "Unlearn Loss Diff": 
        lambda x: x["after"]["loss"] - x["before"]["loss"],
        "Scaled Unlearn Loss Diff": 
        lambda x: x["after"]["loss"] - x["before"]["loss"],
        "LOSS after": 
        lambda x: -x["after"]["loss"],
        "Gradient norm": 
        lambda x: -np.linalg.norm(x["before"]["grads"]),
        "Gradient norm change": 
        lambda x: np.linalg.norm(x["after"]["grads"]) - np.linalg.norm(x["before"]["grads"]),
        "Weight change norm": 
        lambda x: np.linalg.norm(x["before"]["params"] - x["after"]["params"]),
        "Pred entropy change": 
        lambda x: entropy(x["after"]["pred"]) - entropy(x["before"]["pred"]),
    }

    # Scatter plot for two features
    plt.clf()
    plt.scatter(
        np.array([-attacks["LOSS"](x) for x in signals_out]),
        np.array([attacks["Unlearn Loss Diff"](x) for x in signals_out]),
        alpha=0.75,
        label="out",
        s=6,
        marker="o",
    )
    plt.scatter(
        np.array([-attacks["LOSS"](x) for x in signals_in]),
        np.array([attacks["Unlearn Loss Diff"](x) for x in signals_in]),
        alpha=0.75,
        label="in",
        s=6,
        marker="x",
    )
    plt.xlabel("Loss (before)")
    plt.ylabel("Loss (diff)")
    plt.xlim([0, 2])
    plt.ylim([1, 4])
    plt.legend()
    plt.savefig("./scatter.png")
    plt.clf()

    for name, lamb in attacks.items():
        in_pred = np.array([lamb(x) for x in signals_in])
        out_pred = np.array([lamb(x) for x in signals_out])
        total_preds = np.concatenate((out_pred, in_pred))
        total_labels = [0] * len(out_pred) + [1] * len(in_pred)
        fpr, tpr, thresholds = roc_curve(total_labels, total_preds)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label="%s (AUC = %0.3f)" % (name, roc_auc))

    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("TPR")
    plt.xlabel("FPR")
    plt.savefig("./evaluate.png")
    
    # Also save low TPR/FPR region
    plt.xlim([1e-5, 1e0])
    plt.ylim([1e-5, 1e0])
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("./evaluate_lowfpr.png")


if __name__ == "__main__":
    main()
