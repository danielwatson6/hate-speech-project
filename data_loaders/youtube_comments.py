"""Ambiguity data loader."""

from collections import Counter
import os

import pandas as pd
import tensorflow as tf

import boilerplate as tfbp

num_files = 119


def make_csv_dataset(path):
    return tf.data.experimental.make_csv_dataset(
        path, 1, num_epochs=1, shuffle=False, select_columns=['content']
    )


@tfbp.default_export
class YouTube(tfbp.DataLoader):
    default_hparams = {
        "batch_size": 32,
        "punctuation": True,
        "lowercase": True,
        "max_seq_len": None,
        "num_examples": 10000,
    }

    def call(self):
        path = os.path.join(os.environ["DATASETS"], "youtube_right")
        channels = list(os.listdir(path))
        channel_paths = []
        for channel in channels:
            channel_paths.append(os.path.join(path, channel))

        filepath_dataset = make_csv_dataset(channel_paths)

        dataset = filepath_dataset.interleave(
            cycle_length=default_hparams.batch_size, #batch size of 32
            block_length=num_files, # cur num of files is 119 
        )

        if self.hparams.num_examples:
            dataset = dataset.take(self.hparams.num_examples)
        
        dataset = dataset.batch(self.hparams.batch_size)
        dataset = dataset.map(self._dict_to_tensor)
        return dataset.prefetch(1)

    def _dict_to_tensor(self, batch):
        batch = batch["content"]

        if not self.hparams.punctuation:
            batch = tf.strings.regex_replace(batch, "[\.,;:-]", "")
        if self.hparams.lowercase:
            batch = tf.strings.lower(batch)

        batch = tf.strings.split(batch).to_tensor(default_value="<pad>")
        if self.hparams.max_seq_len:
            batch = batch[:, self.hparams.max_seq_len]
        return batch