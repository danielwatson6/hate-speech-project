import os
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def get_ambiguity(save_dir, data_loader):
    """Run the `ambiguity` method for the specified pretrained model and data loader."""
    # Capture the stdout into a variable. 0 max_seq_len will avoid truncation.
    proc = subprocess.Popen(
        f"python run.py ambiguity {save_dir} {data_loader} --max_seq_len=0".split(),
        stdout=subprocess.PIPE,
    )
    output = proc.communicate()[0].decode("utf8")
    # Now parse the printed values.
    ragged = []
    for line in output.split("\n"):
        ragged.append([float(x) for x in line.split(", ")])
    return ragged


def plot_comparison(x1, x2, xlabel="", ylabel=""):
    pcc = pearsonr(x1, x2)
    scc = spearmanr(x1, x2)

    plt.title(
        "Ambiguities (pcc={:.4f} (p={:.4f}) scc={:.4f} (p={:.4f}))".format(*(pcc + scc))
    )
    plt.plot(x1, x2, "o", color="black", markersize=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join("plots", f"{xlabel}_vs_{ylabel}.png"))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python -m scripts.ambiguity.compare [data_loader] [model_1] ... "
            "[model_n]\n(no need to add wordnet model)"
        )
        exit()

    data_loader = sys.argv[1]
    save_dirs = sys.argv[2:]

    ambiguities = [get_ambiguity("wordnet ambiguity_wn", data_loader)]
    for save_dir in save_dirs:
        ambiguities.append(get_ambiguity(save_dir, data_loader))

    axis_labels = ["wordnet"] + save_dirs

    if data_loader == "ambiguity":
        # ambiguity for language model and wordnet, ambiguities array 3D, per word at inner most level, per sentence, outermost level is the language model
        df = pd.read_csv(os.path.join("data", "ambiguity.clean.csv"))

        human_ambiguities = []
        for sequence in df.iterrows():
            index = sequence["index"]
            human_ambiguity = sequence["rating"]
            human_ambiguities.append(human_ambiguity)

        model_ambiguities = []
        for i, model in enumerate(ambiguities):
            model_ambiguities.append([])
            for j, sentence in enumerate(model):
                index = df.iloc[j]["index"]
                ambiguity = sentence[index]
                model_ambiguities[i].append(ambiguity)

        for mambg, ylabel in zip(model_ambiguities, axis_labels):
            plot_comparison(human_ambiguities, mambg, xlabel="human", ylabel=ylabel)

    # Correlation plots between all the distinct pairs of models.
    for i, x1 in enumerate(ambiguities):
        for j, x2 in enumerate(ambiguities[i + 1 :], i + 1):
            x1 = [x for sentence in x1 for x in sentence]
            x2 = [x for sentence in x2 for x in sentence]

            plot_comparison(x1, x2, xlabel=axis_labels[i], ylabel=axis_labels[j])
