import os
import random
import re

import pandas as pd

import utils


if __name__ == "__main__":
    # 37,897 examples from stormfront and twitter datasets
    num_channels = 20
    num_videos_per_channel = 37897 // 20

    path = os.path.join(os.environ["DATASETS"], "youtube_right")
    channels = list(os.listdir(path))
    for channel in channels:
        channel_files = [c for c in os.listdir(path) if c.startswith(channel)]
        for cf in channel_files:
            df = pd.read_csv(os.path.join(path, cf))
            print(df)
