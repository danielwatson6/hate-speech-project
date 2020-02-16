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

def parse_fn(line):
    defs = [0.0] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[14])
    tf.print(x)
    return x


def make_csv_dataset(path):
    return tf.data.experimental.make_csv_dataset(
        path, 1, num_epochs=1, shuffle=False, select_columns=14
    )


if __name__ == "__main__":
    path = os.path.join(os.environ["DATASETS"], "youtube_right")
    channels = list(os.listdir(path))
    channel_paths = []
    for channel in channels:
        channel_paths.append(os.path.join(path, channel))

    filepath_dataset = tf.data.Dataset.list_files(channel_paths, shuffle=False)

    dataset = filepath_dataset.interleave(
        make_csv_dataset,
        cycle_length=32,
        block_length=119,
    )
    for x in dataset:
        print(x)