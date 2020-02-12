import os
import random
import re
import tensorflow as tf
import numpy as np 
import pandas as pd

import utils


if __name__ == "__main__":
    path = os.path.join(os.environ["DATASETS"], "youtube_right")
    channels = list(os.listdir(path))
  
    for channel in channels:    
        channel_files = [c for c in os.listdir(path) if c.startswith(channel)]
        dataset = tf.data.Dataset.from_tensor_slices(channel_files)
        print(dataset)
        

















    
    
