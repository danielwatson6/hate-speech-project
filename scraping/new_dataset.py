from csv import DictWriter
from datetime import datetime
import os

import pandas as pd
import requests


API_KEY = None

COMMENT_KEYS = (
    "id",
    "parent_comment_id",
    "video_id",
    "op_channel_id",
    "like_count",
    "content",
    "date_posted",
    "date_scraped",
)
VIDEO_KEYS = (
    "id",
    "channel_id",
    "title",
    "like_count",
    "dislike_count",
    "view_count",
    "description",
    "content",
    "date_posted",
    "date_scraped",
)
CHANNEL_KEYS = (
    "id",
    "likes_playlist_id",
    "uploads_playlist_id",
    "username",
    "title",
    "description",
    "country",
    "view_count",
    "comment_count",
    "subscriber_count",
    "date_created",
    "date_scraped",
)


def get_missing_comment_data(comment):
    """Missing keys:

    parent_comment_id
    op_channel_id
    like_count

    """
    response = requests.get(
        "https://www.googleapis.com/youtube/v3/comments",
        params=dict(
            id=comment["id"],
            part="snippet",
            key=API_KEY,
            textFormat="plainText",
        ),
    ).json()

    # TODO: better error handling
    if "items" not in response or len(response["items"]) == 0:
        print(response)
        raise KeyError("No items in response")
    response = response["items"][0]

    if "parent_id" in response["snippet"]:
        comment["parent_comment_id"] = response["snippet"]["parent_id"]
    else:
        comment["parent_comment_id"] = ""
    
    try:
        comment["op_channel_id"] = response["snippet"]["authorChannelId"]["value"]
        comment["like_count"] = response["snippet"]["likeCount"]
        comment["date_scraped"] = datetime.now().isoformat()
    except KeyError:
        print(response)
        raise


def get_missing_video_data(video):
    """Missing keys:

    channel_id
    like_count
    dislike_count
    view_count

    """
    response = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params=dict(
            id=video["id"],
            part="snippet,statistics",
            key=API_KEY,
            textFormat="plainText",
        ),
    ).json()

    # TODO: better error handling
    if "items" not in response or len(response["items"]) == 0:
        print(response)
        raise KeyError("No items in response")
    response = response["items"][0]
    
    try:
        video["channel_id"] = response["snippet"]["channelId"]
        video["like_count"] = response["statistics"]["likeCount"]
        video["dislike_count"] = response["statistics"]["dislikeCount"]
        video["view_count"] = response["statistics"]["viewCount"]
        video["date_scraped"] = datetime.now().isoformat()
    except KeyError:
        print(response)
        raise


def get_missing_channel_data(channel):
    """Missing keys:

    likes_playlist_id
    uploads_playlist_id
    video_channel_id
    title
    description
    country
    view_count
    comment_count
    subscribe_count
    date_created

    """

    channel["date_scraped"] = datetime.now().isoformat()


def print_row(row, keys):
    for key in keys:
        text = str(row[key])
        if len(text) > 12:
            text = text[:12] + "..."
        print(text + "\t", end="")
    print()


if __name__ == "__main__":
    with open(os.path.join("scraping", "api_key")) as f:
        API_KEY = f.read().strip()

    path_orig = os.path.join(os.environ["DATASETS"], "youtube_right")
    path_new = os.path.join("data", "youtube_new")

    if not os.path.exists(path_new):
        os.makedirs(path_new)

    file_comments = open(os.path.join(path_new, "comments.csv"), "w", newline="")
    writer_comments = DictWriter(file_comments, COMMENT_KEYS)
    writer_comments.writeheader()
    file_videos = open(os.path.join(path_new, "videos.csv"), "w", newline="")
    writer_videos = DictWriter(file_comments, VIDEO_KEYS)
    writer_videos.writeheader()

    # Keep a dict to query channel by ids without repetition.
    channels = {}

    for filename in os.listdir(path_orig):
        rows = pd.read_csv(os.path.join(path_orig, filename)).iterrows()
        count = 0

        for row in rows:
            count += 1
            if count == 1:
                continue  # skip csv header
            row = row[1]

            if int(row["video"]):
                video = {
                    "id": row["video_id"],
                    "title": row["video_title"],
                    "description": row["video_snippet"],
                    "content": row["content"],
                    "date_posted": row["date_posted"],
                }

                try:
                    get_missing_video_data(video)
                except KeyError:
                    continue

                if video["channel_id"] not in channels:
                    channels[video["channel_id"]] = row["video_op"]

                writer_videos.writerow(video)
                print(count)

            else:
                comment = {
                    "id": row["index"],
                    "video_id": row["video_id"],
                    "content": row["content"],
                    "date_posted": row["date_posted"],
                }
                
                try:
                    get_missing_comment_data(comment)
                except KeyError:
                    continue

                if comment["op_channel_id"] not in channels:
                    if int(row["reply"]):
                        channels[comment["op_channel_id"]] = row["reply_op"]
                    else:
                        channels[comment["op_channel_id"]] = row["comment_op"]
                writer_comments.writerow(comment)
                print(count)

    file_comments.close()
    file_videos.close()
    print("Successfully created videos.csv and comments.csv files.")

    # file_channels = open(os.path.join(path_new, "channels.csv"), "w", newline="")
    # writer_channels = DictWriter(file_comments, CHANNEL_KEYS)
    # writer_channels.writeheader()

    # for channel_id, username in channels.items():
    #     channel = {"id": channel_id, "username": username}
    #     get_missing_channel_data(channel)
    #     writer_channels.writerow(channel)

    # file_channels.close()
