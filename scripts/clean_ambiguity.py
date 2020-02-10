from csv import DictWriter
import json
import os
import re
import string

from nltk.tokenize import word_tokenize
import pandas as pd


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

    keys = ["sentence", "index", "rating"]

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
                # clean data
                row_out = {k: 0 for k in keys}
                # row_out["id"] = row["articleId"]
                # row_out["term"] = row["term"]
                # row_out["sentence"] = row["sentenceText"]

                # replace special characters and numbers
                sentence = row["termInSentence"]
                sentence = re.sub(r"[^A-Za-z<>'\.,;:-]", " ", sentence)
                sentence = re.sub(r"[0-9]+(?:\S[0-9]+)*", " <num> ", sentence)

                tokenized_sentence = word_tokenize(sentence)
                html_tag = tokenized_sentence.index(">")
                # remove first html tags
                del tokenized_sentence[html_tag - 5 : html_tag + 1]
                term_index = html_tag - 5  # term index after removal of first html tag
                # delete trailing </b>
                del tokenized_sentence[term_index + 1 : term_index + 4]
                row_out["sentence"] = " ".join(tokenized_sentence)
                row_out["index"] = term_index
                row_out["rating"] = row["rating_avg"]

                writer.writerow(row_out)

        del data


if __name__ == "__main__":
    original_data()
