"""Scraper for YouTube comment threads."""

# https://2.python-requests.org//en/master/
import requests


COMMENT_THREADS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

# Do not share to prevent unauthorized usage of our API quota.
API_KEY = "AIzaSyAGG5Z-XzbgBj9PSw_U4KDVEdsLRhCQSVU"


def get_comments(comment_id, next_page_token=None, accumulator=None):
    payload = dict(
        commentId=comment_id,
        part="id,snippet",
        key=API_KEY,
        maxResults=100,  # this is the maximum the API allows
        textFormat="plainText",
    )
    if next_page_token is not None:
        payload["nextPageToken"] = next_page_token

    response = requests.get(COMMENT_THREADS_URL, params=payload).json()

    next_page_token = None
    if "nextPageToken" in response:
        next_page_token = response["nextPageToken"]

    if accumulator is None:
        accumulator = []

    # Only keep comments with replies.
    for item in response["items"]:
        if item["snippet"]["totalReplyCount"] > 0:
            accumulator.append(item["snippet"]["topLevelComment"]["id"])

    return accumulator
    # if next_page_token is None:
    #     return accumulator
    # return get_threads_by_video_id(
    #     video_id, next_page_token=next_page_token, accumulator=accumulator
    # )


if __name__ == "__main__":
    print(get_top_level_comments("imGbhF2AEPw"))
