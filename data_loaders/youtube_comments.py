"""Ambiguity data loader."""

from collections import Counter
import os

import pandas as pd
import tensorflow as tf

import boilerplate as tfbp

num_files = 119

@tfbp.default_export
class Ambiguity(tfbp.DataLoader):
    default_hparams = {
        "batch_size": 32,
        "punctuation": True,
        "lowercase": True,
        "max_seq_len": None,
    }
    def make_csv_dataset(path):
        return tf.data.experimental.make_csv_dataset(path, batch_size=1, num_epochs=1, select_columns=14)

    def call(self):
        path = os.path.join(os.environ["DATASETS"], "youtube_right")
        channels = list(os.listdir(path))
        channel_paths = []
        for channel in channels:
            channel_paths.append(os.path.join(path, channel))

        filepath_dataset = tf.data.Dataset.list_files(channel_paths)
    
        dataset = filepath_dataset.interleave(
            make_csv_dataset,
            cycle_length=default_hparams.batch_size,
            block_length=num_files)
        
        # dataset = tf.data.experimental.make_csv_dataset(
        #     os.path.join("data", "ambiguity.clean.csv"),
        #     self.hparams.batch_size,
        #     num_epochs=1,
        #     shuffle=False,
        #     num_rows_for_inference=None,
        # )

        dataset = dataset.map(self._dict_to_tensor)
        return dataset.prefetch(1)

    def _dict_to_tensor(self, batch):
        batch = batch["sentence"]

        if not self.hparams.punctuation:
            batch = tf.strings.regex_replace(batch, "[\.,;:-]", "")
        if self.hparams.lowercase:
            batch = tf.strings.lower(batch)

        batch = tf.strings.split(batch).to_tensor(default_value="<pad>")
        if self.hparams.max_seq_len:
            batch = batch[:, self.hparams.max_seq_len]
        return batch
