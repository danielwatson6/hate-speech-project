from argparse import ArgumentParser
import json
import os
import sys

import tensorflow as tf

import boilerplate


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(
            "Usage: python run.py [method] [save_dir] [model] [data_loader] "
            "[hyperparameters...]",
            file=sys.stderr,
        )
        exit(1)

    # TODO: make it possible to infer the model and data_loader fields by reading the
    # runpy.json file.
    Model = getattr(__import__("models." + sys.argv[3]), sys.argv[3])
    DataLoader = getattr(__import__("data_loaders." + sys.argv[4]), sys.argv[4])

    parser = ArgumentParser()
    parser.add_argument("method", type=str)
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
    kwargs = {k: v for k, v in FLAGS._get_kwargs()}

    del kwargs["model"]
    del kwargs["save_dir"]
    del kwargs["data_loader"]

    model = Model(os.path.join("experiments", FLAGS.save_dir), **kwargs)
    dataset = DataLoader(**kwargs)

    try:
        model.restore()
    except Exception:
        model.save()
        with open(os.path.join("experiments", FLAGS.save_dir, "runpy.json"), "w") as f:
            json.dump({"model": FLAGS.model, "data_loader": FLAGS.data_loader}, f)

    getattr(model, FLAGS.method)(dataset)
