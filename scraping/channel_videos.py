import os
import sys
import time

import requests


API_KEY = None  # set in `main` by reading secrets file


# TODO: more direct way to get videos from channel:
# https://developers.google.com/youtube/v3/docs/search/list
#
# via channelId, type=channel and pageToken to recurse every 500 videos.


def scrape_channel_videos(channel_id):
    # First, we need to get a ChannelSection resource that has the postedVideos type:
    # https://developers.google.com/youtube/v3/docs/channelSections
    #
    # Then, using that channel section id, we can fetch the videos:
    # https://developers.google.com/youtube/v3/docs/channelSections/list
    response = requests.get(
        "https://www.googleapis.com/youtube/v3/channelSections",
        params=dict(
            id=",".join(ids), part="id,snippet", key=API_KEY, textFormat="plainText",
        ),
    ).json()

    # Quota error.
    if "items" not in response:
        print(
            "quota error was likely encountered, will try again in 5min",
            file=sys.stderr,
        )
        time.sleep(300)
        return scrape_channel_videos(channel_id)

    channel_section_id = None
    for channel_section in response["items"]:
        if channel_section["snippet"]["type"] == "postedVideos":
            channel_section_id = channel_section["id"]
            break

    return get_posted_videos(channel_section_id)


def get_posted_videos(channel_section_id):
    response = requests.get(
        "https://www.googleapis.com/youtube/v3/channelSections",
        params=dict(
            id=",".join(ids), part="id,snippet", key=API_KEY, textFormat="plainText",
        ),
    ).json()


def write_ids(video_ids):
    with open(csv_path, mode="a") as f:
        for id_ in video_ids:
            f.write(f"{id_}\n")


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
                for video in scrape_video_batch(ids):
                    write_row(video)
                ids = []
