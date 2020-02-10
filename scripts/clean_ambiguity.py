from csv import DictWriter
import json
import os
import re
import string

from nltk.tokenize import word_tokenize
import pandas as pd


# we have to remove the commas (tozenization)


def original_data():
    # TODO: change where it is reading from
    data = pd.read_csv("../data/ambiguity.csv")
    # data = pd.read_csv(os.path.join("data", "ambiguity.csv"))

    # drop rejected rows
    index_names = data[data["rejected"] == 1].index
    data.drop(index_names, inplace=True)

    # columns to drop:
    to_drop = [
        "block_num",
        "question_num",
        "termInSentence",
        "sampleEnt_category",
        "sampleEnt_global",
        "weight",
        "item_hash",
        "mean_time",
        "worker_anon",
        " ",
        "rejected",
    ]
    data = data.drop(to_drop, axis=1)

    # get means of ratings for each articleId
    means = data.groupby(["articleId"]).mean().round(1)
    means.rename(columns={"rating": "rating_avg"}, inplace=True)
    means.reset_index()
    clean_data = pd.merge(means, data, on="articleId")
    clean_data.drop_duplicates(subset="articleId", inplace=True)
    clean_data = clean_data.drop("rating", axis=1)

    keys = ["id", "term", "sentence", "rating"]

    # write into a new file
    # TODO: change path
    with open("../data/ambiguity.clean.csv", "w") as f:
        # with open(os.path.join("data", "ambiguity.clean.csv"), "w") as f:
        writer = DictWriter(f, keys)
        writer.writeheader()
        line_count = 0  # used to remove column names

        for row in clean_data.iterrows():
            row = row[1]
            if line_count == 0:
                line_count += 1
            else:
                row_out = {k: 0 for k in keys}
                row_out["id"] = row["articleId"]
                row_out["term"] = row["term"]
                row_out["sentence"] = row["sentenceText"]
                row_out["rating"] = row["rating_avg"]
                writer.writerow(row_out)

        del data


if __name__ == "__main__":
    original_data()
