from csv import DictWriter
import json
import os
import re
import string

from nltk.tokenize import word_tokenize
import pandas as pd


def original_data():
    with open(os.path.join("data", "twitter_mf", "MFTC_V4.json")) as f:
        orig = json.load(f)
    for corpus in orig:
        if corpus["Corpus"] == "Davidson":
            continue
        for tweet_meta in corpus["Tweets"]:
            yield tweet_meta
    for corpus in orig:
        if corpus["Corpus"] != "Davidson":
            continue
        for tweet_meta in sorted(corpus["Tweets"], key=lambda x: int(x["tweet_id"])):
            yield tweet_meta


def clean_tweet(tweet):
    tweet = tweet.lower().strip()
    # Remove URLs, usernames and special characters
    tweet = re.sub(r"http\S+", " ", tweet)
    tweet = re.sub(r"@\S+", "@ ACCOUNT", tweet)
    tweet = re.sub(r"[0-9]+(?:,\.[0-9]+)+", "N", tweet)
    tweet = re.sub(f"[^A-Za-z{string.punctuation}]", " ", tweet)
    return " ".join(word_tokenize(tweet))


if __name__ == "__main__":

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
        i, tweet_meta = None, None

        # index,subversion,authority,cheating,fairness,harm,care,betrayal,loyalty,purity,degradation,non_moral,tweet
        scraped = pd.read_csv(os.path.join("data", "twitter_mf.csv"))
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

                row_out["non_moral"] = 0
                Z = sum([v ** 2 for v in row_out.values()]) ** 0.5
                if Z != 0:
                    row_out = {k: v / Z for k, v in row_out.items()}

            row_out["id"] = tweet_meta["tweet_id"]
            row_out["tweet"] = clean_tweet(row["tweet"])

            writer.writerow(row_out)

        del scraped
        davidson = pd.read_csv(os.path.join("data", "twitter.csv"))
        i0 = i + 1

        for _, tweet_meta in it:
            i = int(tweet_meta["tweet_id"])
            row = davidson.iloc[i]
            row_out = {k: 0 for k in keys}

            for annotator in tweet_meta["annotations"]:
                anns = annotator["annotation"].split(",")
                for ann in anns:
                    if ann in ["nm", "non-moral"]:
                        row_out["non_moral"] += 1
                    elif ann == "subversion":
                        row_out["authority"] -= 1
                    elif ann == "cheating":
                        row_out["fairness"] -= 1
                    elif ann == "harm":
                        row_out["care"] -= 1
                    elif ann == "betrayal":
                        row_out["loyalty"] -= 1
                    elif ann == "degradation":
                        row_out["purity"] -= 1
                    else:
                        row_out[ann] += 1

            if int(row_out["non_moral"]) > len(tweet_meta["annotations"]) / 2:
                row_out = {k: 0 for k in keys}
                row_out["non_moral"] = 1
            else:
                row_out["non_moral"] = 0
                Z = sum([v ** 2 for v in row_out.values()]) ** 0.5
                if Z != 0:
                    row_out = {k: v / Z for k, v in row_out.items()}

            row_out["id"] = i
            row_out["tweet"] = clean_tweet(row["tweet"])

            writer.writerow(row_out)
