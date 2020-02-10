"""Ambiguity data loader."""

from collections import Counter
import os

import pandas as pd
import tensorflow as tf

import boilerplate as tfbp


@tfbp.default_export
class Ambiguity(tfbp.DataLoader):
    default_hparams = {
        "batch_size": 32,
        "punctuation": True,
        "lowercase": True,
    }

    def call(self):
        dataset = tf.data.experimental.make_csv_dataset(
            os.path.join("data", "ambiguity.clean.csv"),
            self.hparams.batch_size,
            num_epochs=1,
            shuffle=False,
            num_rows_for_inference=None,
        )
        dataset = dataset.map(self._dict_to_tensor)
        return dataset.prefetch(1)

    def _dict_to_tensor(self, batch):
        batch = batch["sentence"]

        if not self.hparams.punctuation:
            batch = tf.strings.regex_replace(batch, "[\.,;:-]", "")
        if self.hparams.lowercase:
            batch = tf.strings.lower(batch)

        return tf.strings.split(batch).to_tensor(default_value="<pad>")
