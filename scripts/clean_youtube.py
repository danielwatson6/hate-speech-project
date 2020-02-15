import os
import random
import re
import tensorflow as tf
import numpy as np 
import pandas as pd

import utils

n_inputs = 16
n_readers = 5

def parse_fn(line): 
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[14])
    tf.print(x)
    return x


def make_csv_dataset(path):
    ...


if __name__ == "__main__":
    path = os.path.join(os.environ["DATASETS"], "youtube_right")
    channels = list(os.listdir(path))
    channel_paths = []
    for channel in channels:
      channel_paths.append(os.path.join(path, channel))

    filepath_dataset = tf.data.Dataset.list_files(channel_paths)
    print(filepath_dataset)
    for x in filepath_dataset:
        print(x)

    dataset = filepath_dataset.interleave(
        lambda filepath: tf.data.experimental.make_csv_dataset(filepath),
        cycle_length=n_readers)
    dataset = dataset.shuffle(5)
    print(dataset)
    for x in dataset:
        print(x)
    
    # dataset = dataset.map(parse_fn,num_parallel_calls=2)

     

          














    
    
