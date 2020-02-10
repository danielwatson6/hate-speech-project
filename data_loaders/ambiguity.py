"""Ambiguity data loader."""

from collections import Counter
import os

import pandas as pd
import tensorflow as tf

import boilerplate as tfbp


@tfbp.default_export
class ambiguity(tfbp.DataLoader):
    default_hparams = {"batch_size": 32, "max_seq_len": 40}

    def call(self):
        # dataset not used for training

        # TODO: word_to_id and id_to_word as attributes... how to do this without vocab file?

        return self._make_dataset("ambiguity_final.csv", shuffle=10000)

    def _batch_to_ids(self, batch):
        pass

    def _make_dataset(self, path, shuffle=None):
        # TODO: change path
        path = "../data/ambiguity.clean.csv"
        # path =  "os.path.join('data', 'clean_ambiguity.csv')"
        dataset = tf.data.experimental.make_csv_dataset(
            path,  # or whatever you name it
            self.hparams.batch_size,
            num_epochs=1,
            shuffle=False,
            num_rows_for_inference=None,
        )
        dataset = dataset.map(self._dict_to_pair)
        return dataset.prefetch(1)

    def _dict_to_pair(self, batch):
        sentence = batch["sentence"]
        sentence = tf.strings.split(sentence).to_tensor(default_value="<pad>")
        sentence = self._word_to_id.lookup(sentence)  # TODO: word_to_id

        label = batch["rating"]
        label = tf.cast(label, tf.float32)

        return sentence, label
