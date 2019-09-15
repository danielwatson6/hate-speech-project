from csv import DictWriter
import json
import os.path
import time
import traceback

from twython import Twython
from twython.exceptions import TwythonRateLimitError


consumer_key = ""
consumer_secret = ""
access_token_key = ""
access_token_secret = ""


def get_tweet_text(tweet_id):
    try:
        twitter = Twython(
            consumer_key, consumer_secret, access_token_key, access_token_secret
        )
        tweet = twitter.show_status(id=tweet_id)
        return tweet["text"]
    except TwythonRateLimitError:
        print("Quota exceeded. Will try again in 5min.")
        time.sleep(300)
        return get_tweet_text(tweet_id)


def scrape():

    with open(os.path.join("data", "twitter_mf", "MFTC_V4.json")) as f:
        data = json.load(f)

    output_keys = [
        "index",
        "subversion",
        "authority",
        "cheating",
        "fairness",
        "harm",
        "care",
        "betrayal",
        "loyalty",
        "purity",
        "degradation",
        "non_moral",
        "tweet",
    ]

    with open(os.path.join("data", "twitter_mf.csv"), "w") as f:
        writer = DictWriter(f, output_keys)
        writer.writeheader()

        i = 0
        for corpus in data:

            # Skip over the corpus without tweet ids.
            if corpus["Corpus"] == "Davidson":
                continue

            for t in corpus["Tweets"]:
                print(i)
                row = {k: 0 for k in output_keys}
                row["index"] = i
                i += 1

                # Handle deleted/unavailable tweets.
                try:
                    row["tweet"] = get_tweet_text(t["tweet_id"])
                except Exception as e:
                    print(e)
                    continue

                for annotation in t["annotations"]:
                    for label in annotation["annotation"].split(","):
                        if label == "nm" or label == "non-moral":
                            label = "non_moral"
                        row[label] += 1

                writer.writerow(row)


if __name__ == "__main__":

    with open(os.path.join("secrets", "twitter_api.json")) as f:
        api_stuff = json.load(f)
    consumer_key = api_stuff["consumer_key"]
    consumer_secret = api_stuff["consumer_secret"]
    access_token_key = api_stuff["access_token_key"]
    access_token_secret = api_stuff["access_token_secret"]

    scrape()
