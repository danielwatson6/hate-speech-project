import os
import time

import pandas as pd

import utils


if __name__ == "__main__":
    db = utils.firebase()
    data_dir = os.path.join("data", "youtube_new")

    df = pd.read_csv(os.path.join(data_dir, "videos.csv"), dtype=str)
    for index, row in df.iterrows():
        print(index)
        print(row)
        exit()

    df = pd.read_csv(os.path.join(data_dir, "comments.csv"), dtype=str)
    for index, row in df.iterrows():
        doc_ref = db.collections("comments").document(index)
        if timeout_do("get", doc_ref) is None:
            timeout_do("set", doc_ref, value)
