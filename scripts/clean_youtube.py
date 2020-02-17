import os
import random
import re
import tensorflow as tf
import numpy as np
import pandas as pd

import utils

n_inputs = 16
n_readers = 5

default_hparams = {
    "batch_size": 32,
    "punctuation": True,
    "lowercase": True,
    "max_seq_len": None,
    "num_examples": 1000,
}


def parse_fn(x):
    return x["content"]


def make_csv_dataset(path):
    return tf.data.experimental.make_csv_dataset(
        path, 1, num_epochs=1, shuffle=False, select_columns=["content"]
    )

# def filter_fn(item):
#     return item['content']

def _dict_to_tensor(batch):

    # if not self.hparams.punctuation:
    #     batch = tf.strings.regex_replace(batch, "[\.,;:-]", "")
    # if self.hparams.lowercase:
    #     batch = tf.strings.lower(batch)

    batch = tf.strings.split(batch).to_tensor(default_value="<pad>")
    # if self.hparams.max_seq_len:
    #     batch = batch[:, self.hparams.max_seq_len]
    return batch


if __name__ == "__main__":
    path = os.path.join(os.environ["DATASETS"], "youtube_right")
    channels = list(os.listdir(path))
    channel_paths = []
    for channel in channels:
        channel_paths.append(os.path.join(path, channel))


    filepath_dataset = make_csv_dataset(channel_paths)
    filepath_dataset = filepath_dataset.map(lambda x : x["content"])

    
    dataset = filepath_dataset.interleave(
        lambda string_tensor: tf.data.Dataset.from_tensor_slices(string_tensor),
        cycle_length=32,
        block_length=119,
    )


    dataset = dataset.batch(32)

    for x in dataset:
        print(x)
