from csv import DictWriter
from datetime import datetime
import os
import time

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


def get_missing_comment_data(comments):
    """Missing keys:

    parent_comment_id
    op_channel_id
    like_count

    """
    response = requests.get(
        "https://www.googleapis.com/youtube/v3/comments",
        params=dict(
            id="|".join([c["id"] for c in comments]),
            part="snippet",
            key=API_KEY,
            textFormat="plainText",
        ),
    ).json()

    # Quota error.
    if "items" not in response:
        print("Quota error was likely encountered, will try again in 5min:")
        print(response)
        time.sleep(300)
        return get_missing_comment_data(comments)

    result = []
    for c in comments:
        try:
            if "parent_id" in response["snippet"]:
                c["parent_comment_id"] = response["snippet"]["parent_id"]
            else:
                c["parent_comment_id"] = ""
            c["op_channel_id"] = response["snippet"]["authorChannelId"]["value"]
            c["like_count"] = response["snippet"]["likeCount"]

        except KeyError:
            continue

        c["date_scraped"] = datetime.now().isoformat()
        result.append(c)

    return result


def get_missing_video_data(videos):
    """Missing keys:

    channel_id
    like_count
    dislike_count
    view_count

    """
    response = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params=dict(
            id="|".join([v["id"] for v in videos]),
            part="snippet,statistics",
            key=API_KEY,
            textFormat="plainText",
        ),
    ).json()

    # Quota error.
    if "items" not in response:
        print("Quota error was likely encountered, will try again in 5min:")
        print(response)
        time.sleep(300)
        return get_missing_video_data(videos)

    result = []
    for v in videos:
        try:
            v["channel_id"] = response["snippet"]["channelId"]
            v["like_count"] = response["statistics"]["likeCount"]
            v["dislike_count"] = response["statistics"]["dislikeCount"]
            v["view_count"] = response["statistics"]["viewCount"]

        except KeyError:
            continue

        v["date_scraped"] = datetime.now().isoformat()
        result.append(v)

    return result


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
    with open(os.path.join("scraping", "youtube_api_key")) as f:
        API_KEY = f.read().strip()

    path_orig = os.path.join(os.environ["DATASETS"], "youtube_right")
    path_new = os.path.join("data", "youtube_new")

    if not os.path.exists(path_new):
        os.makedirs(path_new)

    # Line-buffered.
    file_comments = open(
        os.path.join(path_new, "comments.csv"), mode="w", buffering=1, newline=""
    )
    writer_comments = DictWriter(file_comments, COMMENT_KEYS)
    writer_comments.writeheader()
    file_videos = open(
        os.path.join(path_new, "videos.csv"), mode="w", buffering=1, newline=""
    )
    writer_videos = DictWriter(file_comments, VIDEO_KEYS)
    writer_videos.writeheader()

    # Keep a dict to query channel by ids without repetition.
    channels = {}

    for filename in os.listdir(path_orig):
        rows = pd.read_csv(os.path.join(path_orig, filename)).iterrows()
        count = 0
        comment_buf = []
        video_buf = []

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
                video_buf.append(video)

            else:
                comment = {
                    "id": row["index"],
                    "video_id": row["video_id"],
                    "content": row["content"],
                    "date_posted": row["date_posted"],
                }
                comment_buf.append(comment)

            if len(video_buf) == 100:
                video_buf = get_missing_video_data(video_buf)
                for video in video_buf:
                    if video["channel_id"] not in channels:
                        channels[video["channel_id"]] = None
                    writer_videos.writerow(video)
                file_videos.flush()
                os.fsync(file_videos.fileno())
                video_buf = []

            elif len(comment_buf) == 100:
                comment_buf = get_missing_comment_data(comment_buf)
                for comment in comment_buf:
                    if comment["op_channel_id"] not in channels:
                        channels[comment["op_channel_id"]] = None
                    writer_comments.writerow(comment)
                file_comments.flush()
                os.fsync(file_comments.fileno())
                comment_buf = []

    # Write remainders in buffer.
    if len(video_buf) > 0:
        video_buf = get_missing_video_data(video_buf)
        for video in video_buf:
            if video["channel_id"] not in channels:
                channels[video["channel_id"]] = None
            writer_videos.writerow(video)
        file_videos.flush()
        os.fsync(file_videos.fileno())

    if len(comment_buf) > 0:
        comment_buf = get_missing_comment_data(comment_buf)
        for comment in comment_buf:
            if comment["op_channel_id"] not in channels:
                channels[comment["op_channel_id"]] = None
            writer_comments.writerow(comment)
        file_comments.flush()
        os.fsync(file_comments.fileno())

    file_comments.close()
    file_videos.close()
    print("Successfully created videos.csv and comments.csv files.")

    with open(os.path.join(path_new, "channel_ids.txt"), "w") as f:
        for channel_id in channels.keys():
            f.write(channel_id + "\n")

    # file_channels = open(os.path.join(path_new, "channels.csv"), "w", newline="")
    # writer_channels = DictWriter(file_comments, CHANNEL_KEYS)
    # writer_channels.writeheader()

    # for channel_id in channels.keys():
    #     channel = {"id": channel_id}
    #     get_missing_channel_data(channel)
    #     writer_channels.writerow(channel)

    # file_channels.close()
