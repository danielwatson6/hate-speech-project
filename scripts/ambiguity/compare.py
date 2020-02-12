import subprocess
import sys

import matplotlib.pyplot as plt

import boilerplate as tfbp


def get_ambiguity(save_dir, data_loader):
    # Run the `ambiguity` method for the specified pretrained model and data loader,
    # capturing the stdout into a variable.
    proc = subprocess.Popen(
        f"python run.py ambiguity {save_dir} {data_loader}".split(),
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
            "[model_n]"
        )
        exit()

    data_loader = sys.argv[1]
    models = sys.argv[2:]

    for model in models:
        ...

    if data_loader == "ambiguity":
        ...
