"""Mean word embedding moral foundations classifier."""

import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp
from models import moral_foundations
import utils


def mish(x):
    """Mish activation function."""
    return x * tf.math.tanh(tf.math.softplus(x))


@tfbp.default_export
class MF_Doc2Vec(moral_foundations):
    default_hparams = {
        **moral_foundations.default_hparams,
        "reduction": "mean",
        "hidden_sizes": [512],
        "dropout": 0.0,
        "optimizer": "adam",
        "learning_rate": 0.001,
    }

    def make_encoder(self):
        encoder = tf.keras.Sequential()

        if self.hparams.reduction == "max":
            encoder.add(tfkl.Lambda(lambda x: tf.reduce_max(x, axis=1)))
        else:
            encoder.add(tfkl.Lambda(lambda x: tf.reduce_mean(x, axis=1)))

        for hidden_size in self.hparams.hidden_sizes:
            encoder.add(tfkl.Dense(hidden_size, activation=mish))
            if self.hparams.dropout > 0:
                encoder.add(tfkl.Dropout(self.hparams.dropout))

        return encoder
