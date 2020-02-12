import os
import random
import re
import tensorflow as tf

import pandas as pd

import utils


if __name__ == "__main__":
    path = os.path.join(os.environ["DATASETS"], "youtube_right")
    channels = list(os.listdir(path))

    youtube_dump = open(os.path.join("../data", "youtube_text_dump.txt"), "w+")
  
    for channel in channels:    
        channel_files = [c for c in os.listdir(path) if c.startswith(channel)]
        for cf in channel_files:
            df = pd.read_csv(os.path.join(path, cf))
            df = df.sample(frac=1).reset_index(drop=True)
            content = df.pop('content')
            print(content.values)
            dataset = tf.data.Dataset.from_tensor_slices(content.values, dtype=tf.string)
            print(dataset)












    
    
