"""RNN-based moral foundations classifier."""

import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp
from models import moral_foundations
import utils


@tfbp.default_export
class MF_RNN(moral_foundations):
    default_hparams = {
        **moral_foundations.default_hparams,
        "rnn_layers": 2,
        "hidden_size": 512,
        "dropout": 0.0,
    }

    def make_encoder(self):
        encoder = tf.keras.Sequential()
        for _ in range(self.hparams.rnn_layers):
            encoder.add(
                tfkl.Bidirectional(
                    tfkl.GRU(
                        self.hparams.hidden_size,
                        dropout=self.hparams.dropout,
                        return_sequences=True,
                    )
                )
            )
        encoder.add(tfkl.Lambda(lambda x: tf.reduce_max(x, axis=1)))
        return encoder
