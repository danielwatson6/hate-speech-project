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
  print(fields)


if __name__ == "__main__":
    path = os.path.join(os.environ["DATASETS"], "youtube_right")
    channels = list(os.listdir(path))
    channel_paths = []
    for channel in channels:
      channel_paths.append(os.path.join(path, channel))
    print(channel_paths)


    filepath_dataset = tf.data.Dataset.list_files(channel_paths, seed=42)
    dataset = filepath_dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath),
        cycle_length=n_readers)
    dataset = dataset.shuffle(5)
    dataset = dataset.map(parse_fn,num_parallel_calls=2)

     

          














    
    
