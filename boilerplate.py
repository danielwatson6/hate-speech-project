"""TensorFlow Boilerplate main module."""

from collections import namedtuple
import json
import os
import sys

import tensorflow as tf


def Hyperparameters(value):
    """Turn a dict of hyperparameters into a nameduple.

    This method will also check if `value` is a namedtuple, and if so, will return it
    unchanged.

    """
    # Don't transform `value` if it's a namedtuple.
    # https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
    t = type(value)
    b = t.__bases__
    if len(b) == 1 and b[0] == tuple:
        fields = getattr(t, "_fields", None)
        if isinstance(fields, tuple) and all(type(name) == str for name in fields):
            return value

    _Hyperparameters = namedtuple("Hyperparameters", value.keys())
    return _Hyperparameters(**value)


class Model(tf.keras.Model):
    """Keras model with hyperparameter parsing and a few other utilities."""

    default_hparams = {}
    _methods = {}

    def __init__(self, save_dir=None, method=None, **hparams):
        super().__init__()
        self._save_dir = save_dir
        self._method = method
        self.hparams = {**self.default_hparams, **hparams}
        self._ckpt = None

        # If the model's hyperparameters were saved, the saved values will be used as
        # the default, but they will be overriden by hyperparameters passed to the
        # constructor as keyword args.
        hparams_path = os.path.join(save_dir, "hparams.json")
        if os.path.isfile(hparams_path):
            with open(hparams_path) as f:
                self.hparams = {**json.load(f), **hparams}
        else:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(hparams_path, "w") as f:
                json.dump(self.hparams._asdict(), f, indent=4, sort_keys=True)

    @property
    def method(self):
        return self._method

    @property
    def hparams(self):
        return self._hparams

    @hparams.setter
    def hparams(self, value):
        self._hparams = Hyperparameters(value)

    @property
    def save_dir(self):
        return self._save_dir

    def save(self):
        if self._ckpt is None:
            self._ckpt = tf.train.Checkpoint(model=self)
        self._ckpt.save(file_prefix=os.path.join(self.save_dir, "model"))

    def restore(self):
        if self._ckpt is None:
            self._ckpt = tf.train.Checkpoint(model=self)
        self._ckpt.restore(tf.train.latest_checkpoint(self.save_dir))


class DataLoader:
    """Data loader class akin to `Model`."""

    default_hparams = {}

    def __init__(self, method=None, **hparams):
        self._method = method
        self.hparams = {**self.default_hparams, **hparams}

    @property
    def method(self):
        return self._method

    @property
    def hparams(self):
        return self._hparams

    @hparams.setter
    def hparams(self, value):
        self._hparams = Hyperparameters(value)

    def __call__(self, *a, **kw):
        return self.call(*a, **kw)

    def call(self):
        raise NotImplementedError


def runnable(f):
    """Mark a method as runnable from `run.py`."""
    setattr(f, "_runnable", True)
    return f


def default_export(cls):
    """Make the class the imported object of the module and compile its runnables."""
    sys.modules[cls.__module__] = cls
    for name, method in cls.__dict__.items():
        if "_runnable" in dir(method) and method._runnable:
            cls._methods[name] = method
    return cls


def get_model(module_str):
    """Import the model in the given module string."""
    return getattr(__import__(f"models.{module_str}"), module_str)


def get_data_loader(module_str):
    """Import the data loader in the given module string."""
    return getattr(__import__(f"data_loaders.{module_str}"), module_str)
