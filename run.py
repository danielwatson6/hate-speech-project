from argparse import ArgumentParser
import json
import inspect
import os
import sys

import tensorflow as tf

import boilerplate


def getcls(module_str):
    head, tail = module_str.split(".")
    return getattr(__import__(f"{head}.{tail}"), tail)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage:\n  New run: python run.py [method] [save_dir] [model] [data_loader]"
            " [hyperparameters...]\n  Existing run: python run.py [method] [save_dir] "
            "[data_loader]? [hyperparameters...]",
            file=sys.stderr,
        )
        exit(1)

    parser = ArgumentParser()
    parser.add_argument("method", type=str)
    parser.add_argument("save_dir", type=str)

    if not os.path.exists("experiments"):
        os.makedirs("experiments")

    # If runpy.json exists, the model and the data loader classes can be inferred and
    # the data loader can be optionally switched. These need to be loaded to get the
    # static default hyperparameters to be read by argparse.
    runpy_json_path = os.path.join("experiments", sys.argv[2], "runpy.json")
    if os.path.exists(runpy_json_path):

        with open(runpy_json_path) as f:
            classes = json.load(f)

        if len(sys.argv) >= 4 and not sys.argv[3].startswith("--"):
            classes["data_loader"] = sys.argv[3]

        Model = getcls("models." + classes["model"])
        try:
            DataLoader = getcls("data_loaders." + classes["data_loader"])
            parser.add_argument("data_loader", type=str)
        except ModuleNotFoundError:
            if len(sys.argv) >= 5 and not sys.argv[4].startswith("--"):
                print(
                    "Warning: model saved at",
                    os.path.join("experiments", sys.argv[2]),
                    f"already points to `models.{classes['model']}`, ignoring...",
                )
                classes["data_loader"] = sys.argv[4]
                DataLoader = getcls("data_loaders." + classes["data_loader"])
                parser.add_argument("model", type=str)
                parser.add_argument("data_loader", type=str)
            else:
                raise

    else:
        Model = getcls("models." + sys.argv[3])
        DataLoader = getcls("data_loaders." + sys.argv[4])

        parser.add_argument("model", type=str)
        parser.add_argument("data_loader", type=str)

        if not os.path.exists(os.path.join("experiments", sys.argv[2])):
            os.makedirs(os.path.join("experiments", sys.argv[2]))

        with open(runpy_json_path, "w") as f:
            json.dump({"model": sys.argv[3], "data_loader": sys.argv[4]}, f)

    args = {}
    saved_hparams = {}
    hparams_json_path = os.path.join("experiments", sys.argv[2], "hparams.json")
    if os.path.exists(hparams_json_path):
        with open(hparams_json_path) as f:
            saved_hparams = json.load(f)
    for name, value in Model.default_hparams.items():
        if name in saved_hparams:
            value = saved_hparams[name]
        args[name] = value
    for name, value in DataLoader.default_hparams.items():
        if name in saved_hparams:
            value = saved_hparams[name]
        args[name] = value

    for name, value in args.items():
        if type(value) in [list, tuple]:
            if not len(value):
                raise ValueError(
                    f"Cannot infer type of hyperparameter `{name}`. Please provide a "
                    "default value with nonzero length."
                )
            parser.add_argument(
                f"--{name}", f"--{name}_", nargs="+", type=type(value[0]), default=value
            )
        else:
            parser.add_argument(f"--{name}", type=type(value), default=value)

    FLAGS = parser.parse_args()
    kwargs = {k: v for k, v in FLAGS._get_kwargs()}

    for k in ["model", "save_dir", "data_loader"]:
        if k in kwargs:
            del kwargs[k]

    model = Model(os.path.join("experiments", FLAGS.save_dir), **kwargs)
    data_loader = DataLoader(**kwargs)

    try:
        model.restore()
    except Exception:
        model.save()
        with open(os.path.join("experiments", FLAGS.save_dir, "runpy.json"), "w") as f:
            json.dump({"model": FLAGS.model, "data_loader": FLAGS.data_loader}, f)

    if FLAGS.method not in Model._methods:
        methods_str = "\n  ".join(Model._methods.keys())
        raise ValueError(
            f"Model does not have a runnable method `{FLAGS.method}`. Methods available:"
            f"\n  {methods_str}"
        )
    Model._methods[FLAGS.method](model, data_loader)
