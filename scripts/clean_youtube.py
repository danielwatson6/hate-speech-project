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
        for cf in channel_files:
            youtube_dump = open(os.path.join("../data", f"{cf}_text_dump.txt"), "w+")
            df = pd.read_csv(os.path.join(path, cf))
            df = df.sample(frac=1).reset_index(drop=True)
            for rec_index, rec in df.iterrows():
                youtube_dump.write(df['content'] + '\n')
            youtube_dump.close()
















    
    
