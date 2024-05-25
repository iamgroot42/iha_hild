"""
    For a given file directory, read all pytorch .ch files and in those dictionaries, extract "acc".
    Average out this value across the models (with +/- std)
"""
import os
import numpy as np
import sys
import torch as ch
from tqdm import tqdm

def read_accs(file_dir):
    # Get all the files
    files = os.listdir(file_dir)
    # Filter out the .ch files
    files = [f for f in files if f.endswith(".pt")]

    # Read the files
    accs = []
    for f in tqdm(files, total=len(files)):
        # Load the file
        d = ch.load(os.path.join(file_dir, f))
        # Extract the accuracy
        accs.append(d["acc"])

    # Average the accuracies
    accs = np.array(accs)
    avg_acc = np.mean(accs)
    std_acc = np.std(accs)
    return avg_acc, std_acc


if __name__ == "__main__":
    file_dir = sys.argv[1]
    avg_acc, std_acc = read_accs(file_dir)
    print("Average Accuracy: %.3f " % avg_acc)
    print("Standard Deviation: %.3f" % std_acc)