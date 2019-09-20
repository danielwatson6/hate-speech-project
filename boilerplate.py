from collections import namedtuple
import json
import os
import sys

import tensorflow as tf


def Hyperparameters(value):
    # Don't transform the value if it's a namedtuple.
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
    default_hparams = {}

    def __init__(self, save_dir=None, method=None, **hparams):
        super().__init__()
        self._save_dir = save_dir
        self._method = method
        self.hparams = {**self.default_hparams, **hparams}
        self._ckpt = None

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


def default_export(x):
    """Decorator to make a class or method the imported object of a module."""
    sys.modules[x.__module__] = x
    return x
