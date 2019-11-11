from csv import DictWriter
from datetime import datetime
import os
import sys
import time

import pandas as pd
import requests


K = 100

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


def get_missing_comment_data(comments, ids_to_skip):
    """Missing keys:

    parent_comment_id
    op_channel_id
    like_count

    """
    ids = [c["id"] for c in comments]
    response = requests.get(
        "https://www.googleapis.com/youtube/v3/comments",
        params=dict(
            id=",".join(ids), part="snippet", key=API_KEY, textFormat="plainText"
        ),
    ).json()

    # Quota error.
    if "items" not in response:
        print("Quota error was likely encountered, will try again in 5min:")
        print(response)
        time.sleep(300)
        return get_missing_comment_data(comments)

    result = []
    for c, res_item in zip(comments, response["items"]):
        try:
            if "parentId" in res_item["snippet"]:
                c["parent_comment_id"] = res_item["snippet"]["parentId"]
            else:
                c["parent_comment_id"] = ""
            c["op_channel_id"] = res_item["snippet"]["authorChannelId"]["value"]
            c["like_count"] = res_item["snippet"]["likeCount"]

        except KeyError as e:
            print(e)
            continue

        c["date_scraped"] = datetime.now().isoformat()
        print(c)
        result.append(c)

    for id_ in ids:
        writeskip(id_, ids_to_skip)

    return result


def get_missing_video_data(videos, ids_to_skip):
    """Missing keys:

    channel_id
    like_count
    dislike_count
    view_count

    """
    ids = [v["id"] for v in videos]
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
        print("Quota error was likely encountered, will try again in 5min:")
        print(response)
        time.sleep(300)
        return get_missing_video_data(videos)

    result = []
    print(response)
    for v, res_item in zip(videos, response["items"]):
        try:
            v["channel_id"] = res_item["snippet"]["channelId"]
            v["like_count"] = res_item["statistics"]["likeCount"]
            v["dislike_count"] = res_item["statistics"]["dislikeCount"]
            v["view_count"] = res_item["statistics"]["viewCount"]

        except KeyError as e:
            print(e)
            continue

        v["date_scraped"] = datetime.now().isoformat()
        result.append(v)

    for id_ in ids:
        writeskip(id_, ids_to_skip)

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


def writerow(row, keys, path):
    print("    " + str(row["content"])[:28])
    with open(path, mode="a", newline="") as f:
        writer = DictWriter(f, keys)
        # Write the header if the file is empty.
        if os.stat(path).st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def writechid(chid, channels):
    chid = chid.strip()
    if chid not in channels:
        channels[chid] = None
        with open(os.path.join("data", "youtube_new", "channel_ids.txt"), "a") as f:
            f.write(chid + "\n")


def writeskip(id_, ids_to_skip):
    id_ = id_.strip()
    if id_ not in ids_to_skip:
        ids_to_skip[id_] = None
        with open(os.path.join("data", "youtube_new", "ids_to_skip.txt"), "a") as f:
            f.write(id_ + "\n")


def repeat(it):
    while True:
        for el in it:
            yield el


if __name__ == "__main__":

    BUFSIZE = 20
    if len(sys.argv) > 1:
        BUFSIZE = int(sys.argv[1])

    with open(os.path.join("secrets", "youtube_api_key")) as f:
        API_KEY = f.read().strip()

    path_orig = os.path.join(os.environ["DATASETS"], "youtube_right")
    path_new = os.path.join("data", "youtube_new")
    if not os.path.exists(path_new):
        os.makedirs(path_new)

    # Avoid re-scraping anything that we TRIED to scrape (even if API returned say 404)
    ids_to_skip = {}

    # Get all the ids to skip.
    for filename in ["comments.csv", "videos.csv", "channels.csv"]:
        filepath = os.path.join(path_new, filename)
        if os.path.isfile(filepath):
            with open(filepath) as f:
                for line in f:
                    ids_to_skip[line.split(",")[0]] = None

    # Get all the ids to skip.
    if os.path.isfile(os.path.join(path_new, "ids_to_skip.txt")):
        with open(os.path.join(path_new, "ids_to_skip.txt")):
            for line in f:
                ids_to_skip[line.strip()] = None

    path_comments = os.path.join(path_new, "comments.csv")
    path_videos = os.path.join(path_new, "videos.csv")

    # Keep a dict to query channel by ids without repetition.
    channels = {}
    if os.path.isfile(os.path.join(path_new, "channel_ids.txt")):
        with open(os.path.join(path_new, "channel_ids.txt")) as f:
            for line in f:
                channels[line.strip()] = None

    files_finished = {}
    for filename in repeat(os.listdir(path_orig)):
        all_files_finished = True
        for filename_ in os.listdir(path_orig):
            if filename_ not in files_finished:
                all_files_finished = False
        if all_files_finished:
            break

        if filename in files_finished:
            continue

        print(filename)
        rows = pd.read_csv(os.path.join(path_orig, filename)).iterrows()

        count = 0
        broke_halfway = False

        comment_buf = []
        video_buf = []

        for row in rows:
            count += 1
            if count == 1:
                continue  # skip csv header
            row = row[1]

            if int(row["video"]) and row["video_id"] not in ids_to_skip:
                video = {
                    "id": row["video_id"],
                    "title": row["video_title"],
                    "description": row["video_snippet"],
                    "content": row["content"],
                    "date_posted": row["date_posted"],
                }
                video_buf.append(video)

            elif row["index"] not in ids_to_skip:
                comment = {
                    "id": row["index"],
                    "video_id": row["video_id"],
                    "content": row["content"],
                    "date_posted": row["date_posted"],
                }
                comment_buf.append(comment)

            if len(video_buf) == BUFSIZE:
                video_buf = get_missing_video_data(video_buf, ids_to_skip)
                print(f"  retrieved {len(video_buf)} videos")
                for video in video_buf:
                    writechid(video["channel_id"], channels)
                    writerow(video, VIDEO_KEYS, path_videos)
                    count += 1
                video_buf = []

            if len(comment_buf) == BUFSIZE:
                comment_buf = get_missing_comment_data(comment_buf, ids_to_skip)
                print(f"  retrieved {len(comment_buf)} comments")
                for comment in comment_buf:
                    writechid(comment["op_channel_id"], channels)
                    writerow(comment, COMMENT_KEYS, path_comments)
                    count += 1
                comment_buf = []

            if count > K:
                broke_halfway = True
                break

        if not broke_halfway:
            files_finished[filename] = None

        # Write remainders in buffer.
        if len(video_buf) > 0:
            video_buf = get_missing_video_data(video_buf, ids_to_skip)
            for video in video_buf:
                writechid(video["channel_id"], channels)
                writerow(video, VIDEO_KEYS, path_videos)

        if len(comment_buf) > 0:
            comment_buf = get_missing_comment_data(comment_buf, ids_to_skip)
            for comment in comment_buf:
                writechid(comment["op_channel_id"], channels)
                writerow(comment, COMMENT_KEYS, path_comments)

    print("Successfully created videos.csv and comments.csv files.")

    # file_channels = open(os.path.join(path_new, "channels.csv"), mode="a", newline="")
    # writer_channels = DictWriter(file_comments, CHANNEL_KEYS)
    # writer_channels.writeheader()

    # for channel_id in channels.keys():
    #     channel = {"id": channel_id}
    #     get_missing_channel_data(channel)
    #     writer_channels.writerow(channel)

    # file_channels.close()
