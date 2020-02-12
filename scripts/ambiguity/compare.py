import os
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def get_ambiguity(save_dir, data_loader):
    # Run the `ambiguity` method for the specified pretrained model and data loader,
    # capturing the stdout into a variable. 0 max_seq_len will avoid truncation.
    proc = subprocess.Popen(
        f"python run.py ambiguity {save_dir} {data_loader} --max_seq_len=0".split(),
        stdout=subprocess.PIPE,
    )
    output = proc.communicate()[0]
    # Now parse the printed values.
    ragged = []
    for line in output.split("\n"):
        ragged.append([float(x) for x in line.split()])
    return ragged


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python -m scripts.ambiguity.compare [data_loader] [model_1] ... "
            "[model_n]\n(no need to add wordnet model)"
        )
        exit()

    data_loader = tfbp.get_data_loader(sys.argv[1])
    save_dirs = sys.argv[2:]

    ambiguities = [get_ambiguity("wordnet ambiguity_wn", data_loader)]
    for save_dir in save_dirs:
        ambiguities.append(get_ambiguity(save_dir, data_loader))

    # if data_loader == "ambiguity":
    #     # ambiguity for language model and wordnet, ambiguities array 3D, per word at inner most level, per sentence, outermost level is the language model
    #     df = pd.read_csv(os.path.join("data", "ambiguity.clean.csv"))

    #     ambiguities_per_word = []

    #     human_ambiguities = []
    #     for sequence in df.iterrows():
    #         index = sequence["index"]
    #         human_ambiguity = sequence["rating"]
    #         human_ambiguities.append(human_ambiguity)

    #     word_net = []
    #     language_model = []
    #     for m, model in enumrate(ambiguities):
    #         model_comparison = []
    #         for i, sentence in enumerate(model):
    #             index = df.iloc[i]["index"]
    #             ambiguity = sentence[index]
    #             if m == 0:

    #             # model_comparison.append()

    #             # compare rating and model rating
    #             ...

    # Correlation plots between all the distinct pairs of models.
    save_dirs = ["wordnet"] + save_dirs
    for i, x1 in enumerate(ambiguities):
        for j, x2 in enumerate(ambiguities[i + 1 :], i + 1):
            x1 = [x for sentence in x1 for x in sentence]
            x2 = [x for sentence in x2 for x in sentence]

            # Correlation coefficients with p values.
            pcc = pearsonr(x1, x2)
            scc = spearmanr(x1, x2)

            plt.title(
                "Ambiguities (pcc={:.4f} (p={:.4f}) scc={:.4f} (p={:.4f}))".format(
                    *(pcc + scc)
                )
            )
            plt.plot(x1, x2, "o", color="black", markersize=2)
            plt.xlabel(save_dirs[i])
            plt.ylabel(save_dirs[j])
            plt.show()
