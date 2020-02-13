import os
import random
import re
import tensorflow as tf
import numpy as np 
import pandas as pd

import utils


def parse_fn(filename): 
  return tf.data.Dataset.range(10)

if __name__ == "__main__":
    path = os.path.join(os.environ["DATASETS"], "youtube_right")
    channels = list(os.listdir(path))
    channel_paths = []
    for channel in channels:
      channel_paths.append(os.path.join(path, channel))
    print(channel_paths)


    files_ds = tf.data.Dataset.from_tensor_slices(channel_paths)
    lines_ds = files_ds.interleave(tf.data.TextLineDataset, cycle_length=20)

    for i, line in enumerate(lines_ds.take(9)):
      if i % 3 == 0:
        print()
      print(line.numpy())

     

          














    
    
