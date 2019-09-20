import os

import arrow
import pandas as pd

import utils


def parse_row(row):
    row = row.to_dict()
    row["like_count"] = int(row["like_count"])
    row["date_posted"] = arrow.get(row["date_posted"], "YYYY_MM_DD_HH_mm_ss").timestamp
    row["date_scraped"] = arrow.get(
        " ".join(row["date_scraped"].split(".")[0].split("T"))
    ).timestamp

    if "dislike_count" in row:
        row["dislike_count"] = int(row["dislike_count"])
    if "view_count" in row:
        row["view_count"] = int(row["view_count"])

    return (row,)


if __name__ == "__main__":
    db = utils.firebase()
    data_dir = os.path.join("data", "youtube_new")

    df = pd.read_csv(os.path.join(data_dir, "videos.csv"), index_col="id", dtype=str)
    for index, row in df.iterrows():
        doc_ref = db.collection("videos").document(index)
        if utils.timeout_do("get", doc_ref).to_dict() is None:
            utils.timeout_do("set", doc_ref, args=parse_row(row))

    df = pd.read_csv(os.path.join(data_dir, "comments.csv"), index_col="id", dtype=str)
    for index, row in df.iterrows():
        doc_ref = db.collection("comments").document(index)
        if utils.timeout_do("get", doc_ref).to_dict() is None:
            utils.timeout_do("set", doc_ref, args=parse_row(row))
