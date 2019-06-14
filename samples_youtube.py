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

    with open(os.path.join("data", "youtube_samples.txt"), "w") as f:
        acc = []
        for channel in channels[:num_channels]:
            channel_files = [c for c in channels if c.startswith(channel)]

            for cf in channel_files:
                df = pd.read_csv(os.path.join(path, cf))
                # 'index', 'video_id', 'video_url', 'video_title', 'channel', 'series',
                # 'video_snippet', 'video', 'comment', 'reply', 'video_op',
                # 'comment_op', 'reply_op', 'date_posted', 'content', 'date_scraped'
                for row in df.iterrows():
                    acc.append(row["content"])

                    if len(acc) >= num_videos_per_channel:
                        break

                if len(acc) >= num_videos_per_channel:
                    break

            if len(acc) >= num_videos_per_channel:
                for comment in acc:
                    f.write(" ".join(utils.tokenize(comment)) + "\n")
                acc = []
                continue
