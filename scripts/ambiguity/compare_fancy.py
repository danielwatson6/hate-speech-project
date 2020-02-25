import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def get_floats(line):
    return line.strip().split(", ")


def get_ambiguity(save_dir):
    """Run the `ambiguity` method for the specified pretrained model."""
    # Capture the stdout into a variable. 0 max_seq_len will avoid truncation.
    command = f"python run.py ambiguity {save_dir} ambiguity --max_seq_len=0"

    # TODO: get this outta here
    if not save_dir.startswith("wordnet"):
        command += " --vocab_path=data/wikitext-2/wiki.vocab.tsv"

    proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output = proc.communicate()[0].decode("utf8")
    # Now parse the printed values.
    ragged = []
    for line in output.split("\n"):
        ragged.append([float(x) for x in get_floats(line) if line])
    return ragged


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.ambiguity.compare [model]")
        exit()

    save_dir = sys.argv[1]

    all_wordnet_ambg = get_ambiguity("wordnet ambiguity_wn")
    all_model_ambg = get_ambiguity(save_dir)

    df = pd.read_csv(os.path.join("data", "ambiguity.clean.csv"))

    human_ambg = []
    wordnet_ambg = []
    model_ambg = []
    for i, sequence in df.iterrows():
        index = sequence["index"]
        human_ambg.append(sequence["rating"])
        wordnet_ambg.append(all_wordnet_ambg[i][index])
        model_ambg.append(all_model_ambg[i][index])

    human_max = 7.0
    wordnet_max = 71.0
    with open(os.path.join("experiments", save_dir, "hparams.json")) as f:
        model_max = np.log(json.load(f)["vocab_size"])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    # human v.s. wordnet
    ax1.set_title(f"human v.s. wordnet")
    ax1.plot(human_ambg, wordnet_ambg, "o", color="black", markersize=2)
    ax1.set_xlim(0.5, human_max + 0.5)
    ax1.set_ylim(0.0, wordnet_max + 5.0)

    # human v.s. language model
    ax2.set_title(f"human v.s. language model ({save_dir})")
    ax2.plot(human_ambg, model_ambg, "o", color="black", markersize=2)
    ax2.set_xlim(0.5, human_max + 0.5)
    ax2.set_ylim(0.0, model_max + 1.0)

    # wordnet v.s. language model
    ax3.set_title(f"wordnet v.s. language model ({save_dir})")
    ax3.plot(wordnet_ambg, model_ambg, "o", color="black", markersize=2)
    ax3.set_xlim(0.0, wordnet_max + 5.0)
    ax3.set_ylim(0.0, model_max + 1.0)

    fig.tight_layout()
    plt.show()
