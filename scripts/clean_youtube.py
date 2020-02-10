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

    channels = list({re.sub(r"([a-z\.]+)[0-9]*", r"\1", f) for f in os.listdir(path)})
    random.shuffle(channels)
    print(channels)
