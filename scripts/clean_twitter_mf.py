from csv import DictWriter
import json
import os

from nltk.tokenize import word_tokenize
import pandas as pd


def original_data():
    with open(os.path.join("data", "twitter_mf", "MFTC_V4.json")) as f:
        orig = json.load(f)
    for corpus in orig:
        if corpus == "Davidson":
            continue
        for tweet_meta in corpus["Tweets"]:
            yield tweet_meta


def clean_tweet(tweet):
    tweet = tweet.lower()
    tweet = " ".join(word_tokenize(tweet))


if __name__ == "__main__":
    # index,subversion,authority,cheating,fairness,harm,care,betrayal,loyalty,purity,degradation,non_moral,tweet
    scraped = pd.read_csv(os.path.join("data", "twitter_mf.csv"))

    keys = [
        "id",
        "authority",
        "fairness",
        "care",
        "loyalty",
        "purity",
        "non_moral",
        "tweet",
    ]

    with open(os.path.join("data", "twitter_mf.clean.csv"), "w") as f:
        writer = DictWriter(f, keys)
        writer.writeheader()

        it = enumerate(original_data())

        for row in scraped.iterrows():
            row = row[1]
            i, tweet_meta = next(it)

            while i < int(row["index"]):
                i, tweet_meta = next(it)

            row_out = {k: 0 for k in keys}

            if int(row["non_moral"]) > len(tweet_meta["annotations"]) / 2:
                row_out["non_moral"] = 1

            else:
                row_out["authority"] = row["authority"] - row["subversion"]
                row_out["fairness"] = row["fairness"] - row["cheating"]
                row_out["care"] = row["care"] - row["harm"]
                row_out["loyalty"] = row["loyalty"] - row["betrayal"]
                row_out["purity"] = row["purity"] - row["degradation"]

                Z = sum([v ** 2 for v in row_out.values()]) ** 0.5
                if Z != 0:
                    row_out = {k: v / Z for k, v in row_out.items()}

            row_out["id"] = tweet_meta["tweet_id"]
            row_out["tweet"] = clean_tweet(row["tweet"])

            writer.writerow(row_out)
