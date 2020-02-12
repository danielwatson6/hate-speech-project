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
  
    for channel in channels:    
        channel_files = [c for c in os.listdir(path) if c.startswith(channel)]
        dataset = tf.data.Dataset.from_tensor_slices(channel_files)
        dataset = dataset.interleave(lambda x: 
            tf.data.TextLineDataset(x).map(parse_fn, num_parallel_calls=1), 
            cycle_length=4, block_length=16) 
        

















    
    
