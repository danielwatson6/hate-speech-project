from csv import DictWriter
import os
import sys
import time

import pandas as pd
import requests


API_KEY = None  # set in `main` by reading secrets file

VIDEO_FIELDS = (
    "id",
    "channel_id",
    "title",
    "description",
    "view_count",
    "like_count",
    "dislike_count",
    "comment_count",
    "content",
    "date_posted",
    "date_scraped",
)


def scrape_video_batch(ids):
    # The snippet property contains channelId, title, description, publishedAt.
    # The statistics property contains viewCount, likeCount, dislikeCount, commentCount.
    # https://developers.google.com/youtube/v3/docs/videos
    response = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params=dict(
            id=",".join(ids),
            part="snippet,statistics",
            key=API_KEY,
            textFormat="plainText",
        ),
    ).json()

    # Quota error.
    if "items" not in response:
        print(
            "quota error was likely encountered, will try again in 5min",
            file=sys.stderr,
        )
        time.sleep(3600)
        return scrape_video_batch(videos)

    return [parse_video(id_, v) for v in zip(ids, response["items"]) if v is not None]


def parse_video(video_id, video_data):
    v = {}
    # TODO: add all the other fields.
    try:
        v["id"] = video_id
        v["channel_id"] = video_data["snippet"]["channelId"]
        v["title"] = video_data["snippet"]["title"]
        v["description"] = video_data["snippet"]["description"]
        v["view_count"] = video_data["statistics"]["viewCount"]
        v["like_count"] = video_data["statistics"]["likeCount"]
        v["dislike_count"] = video_data["statistics"]["dislikeCount"]
        v["content"] = get_video_transcript(video_id)
        v["date_posted"] = video_data["snippet"]["publishedAt"]
        v["date_scraped"] = time.time()
        return v

    except KeyError as e:
        id_ = v["id"]
        print(
            f"skipping scraped video (id={id_}) with missing key `{e}`", file=sys.stderr
        )
        return


def get_video_transcript(video_id):
    # First, we need the caption ID, which can be fetched by an API call:
    # https://developers.google.com/youtube/v3/docs/captions/list
    #
    # Then, that ID can be used to download the transcript by another API call:
    # https://developers.google.com/youtube/v3/docs/captions/download
    response = requests.get(
        "https://www.googleapis.com/youtube/v3/captions",
        params=dict(part="id", videoId=video_id),
    ).json()

    if "items" not in response:
        print(
            "quota error was likely encountered, will try again in 5min",
            file=sys.stderr,
        )
        time.sleep(300)
        return get_video_transcript(video_id)

    caption_id = repsonse["items"][0]["id"]
    return download_video_transcript(caption_id)


def download_video_transcript(caption_id):
    # https://www.3playmedia.com/2015/03/05/caption-format-acronyms-explained/
    # This seems to be the easiest format to clean.
    response = requests.get(
        "https://www.googleapis.com/youtube/v3/captions/id",
        params=dict(id=caption_id, tfmt="vtt"),
    )
    # TODO: call this function again in case of a quota error. But how to detect
    # quota error if the response is not in JSON format?
    #
    # This will be easy to resolve, if the error happens once, we can find out which
    # error it is and do a try/catch.
    return response.text


def write_row(video):
    csv_path = os.path.join("data", "videos.csv")
    with open(csv_path, mode="a", newline="") as f:
        writer = DictWriter(f, VIDEO_FIELDS)
        # Write the header if the file is empty.
        if os.stat(csv_path).st_size == 0:
            writer.writeheader()
        writer.writerow(video)


if __name__ == "__main__":
    BATCH_SIZE = 20  # number of videos to scrape per API request

    with open(os.path.join("secrets", "youtube_api_key")) as f:
        API_KEY = f.read().strip()

    # Collect the video id's that have already been scraped.
    scraped_ids = set()

    csv_path = os.path.join("data", "videos.csv")
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        for i, row in df.iterrows():
            scraped_ids.add(row["id"])

    with open(os.path.join("data", "video_ids.txt")) as f:
        ids = []
        for line in f:
            id_ = line.strip()  # remove newlines / trailing spaces

            if id_ in scraped_ids:
                continue
            ids.append(id_)

            if len(ids) == BATCH_SIZE:
                videos = scrape_video_batch(ids)
                for video in videos:
                    write_row(video)
                print(f"scraped {len(videos)} videos")
                ids = []
