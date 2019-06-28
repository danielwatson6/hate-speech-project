from argparse import ArgumentParser
import os
import sys

import tensorflow as tf

import boilerplate


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python train.py [save_dir] [model] [data_loader] "
            "[hyperparameters...]",
            file=sys.stderr,
        )
        exit(1)

    Model = getattr(__import__("models." + sys.argv[2]), sys.argv[2])
    DataLoader = getattr(__import__("data_loaders." + sys.argv[3]), sys.argv[3])

    parser = ArgumentParser()
    parser.add_argument("save_dir", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("data_loader", type=str)

    args = {}
    for name, value in Model.default_hparams.items():
        args[name] = value
    for name, value in DataLoader.default_hparams.items():
        args[name] = value
    for name, value in args.items():
        parser.add_argument(f"--{name}", type=type(value), default=value)
    FLAGS = parser.parse_args()
    hparams = {k: v for k, v in FLAGS._get_kwargs()}
    del hparams["model"]
    del hparams["save_dir"]
    del hparams["data_loader"]

    model = Model(os.path.join("model_files", FLAGS.save_dir), **hparams)
    dataset = DataLoader(**hparams).data

    try:
        model.restore()
    except:
        model.save()

    model.train(dataset)
