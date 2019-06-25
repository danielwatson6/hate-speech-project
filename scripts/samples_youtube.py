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

    video_file = open(os.path.join("data", "youtube_video_samples.txt"), "w")
    comment_file = open(os.path.join("data", "youtube_comment_samples.txt"), "w")

    acc = []
    for channel in channels[:num_channels]:
        channel_files = [c for c in os.listdir(path) if c.startswith(channel)]

        for cf in channel_files:
            df = pd.read_csv(os.path.join(path, cf))
            # 'index', 'video_id', 'video_url', 'video_title', 'channel', 'series',
            # 'video_snippet', 'video', 'comment', 'reply', 'video_op',
            # 'comment_op', 'reply_op', 'date_posted', 'content', 'date_scraped'
            for row in df.iterrows():
                row = row[1]
                if int(row["video"]) == 1:
                    video_file.write(
                        " ".join(utils.tokenize(str(row["content"]))) + "\n"
                    )
                else:
                    acc.append(row["content"])

                if len(acc) >= num_videos_per_channel:
                    break

            if len(acc) >= num_videos_per_channel:
                break

        if len(acc) >= num_videos_per_channel:
            for comment in acc:
                comment_file.write(" ".join(utils.tokenize(str(comment))) + "\n")
            acc = []
            continue

    video_file.close()
    comment_file.close()
