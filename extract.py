from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
from dotenv import load_dotenv
import os
import time
import random

#load the api key from the environment variable, i.e., the .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

youtube = build("youtube", "v3", developerKey=API_KEY)

def safe_api_call(func, *args, **kwargs):
    """Retry API calls with exponential backoff on errors."""
    max_retries = 5
    wait_time = 2
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except HttpError as e:
            error_reason = ""
            if e.resp and hasattr(e, 'error_details'):
                error_reason = str(e.error_details)
            elif hasattr(e, 'content'):
                error_reason = str(e.content)

            print(f"⚠️ API Error: {error_reason}")

            if "quotaExceeded" in error_reason:
                print("⏳ Quota exceeded — sleeping for 1 hour...")
                time.sleep(3600)  # sleep for 1 hour
            else:
                sleep_for = wait_time * (2 ** attempt) + random.uniform(0, 1)
                print(f"🔄 Retrying in {sleep_for:.1f} seconds...")
                time.sleep(sleep_for)
    print("Max retries reached — skipping.")
    return None


def get_comments(video_id):
    """Fetch comments and their replies for a video."""
    comments_data = []

    request = youtube.commentThreads().list(
        part="snippet,replies",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    while request:
        response = safe_api_call(request.execute)
        if not response:
            break

        for item in response.get("items", []):
            top_comment = item["snippet"]["topLevelComment"]["snippet"]
            comments_data.append({
                "video_id": video_id,
                "author": top_comment.get("authorDisplayName"),
                "comment": top_comment.get("textDisplay"),
                "likes": top_comment.get("likeCount"),
                "published_at": top_comment.get("publishedAt"),
                "reply_to": None  # top-level comment
            })

            # Handle replies
            if "replies" in item:
                for reply in item["replies"].get("comments", []):
                    reply_snippet = reply["snippet"]
                    comments_data.append({
                        "video_id": video_id,
                        "author": reply_snippet.get("authorDisplayName"),
                        "comment": reply_snippet.get("textDisplay"),
                        "likes": reply_snippet.get("likeCount"),
                        "published_at": reply_snippet.get("publishedAt"),
                        "reply_to": top_comment.get("authorDisplayName")
                    })

        request = youtube.commentThreads().list_next(request, response)

    return comments_data