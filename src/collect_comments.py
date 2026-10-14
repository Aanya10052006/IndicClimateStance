
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("YOUTUBE_API_KEY")

MAX_COMMENTS_PER_VIDEO = 500

INVENTORY_FILE = PROJECT_ROOT / "data" / "raw" / "video_inventory.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "raw_comments.csv"

COLUMNS = [
    "comment_id",
    "video_id",
    "language",
    "comment_text",
    "parent_comment_id",
]


def collect_comments(youtube, video_id, language, max_comments):
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        page_size = min(100, max_comments - len(comments))

        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=page_size,
            textFormat="plainText",
            pageToken=next_page_token,
        ).execute()

        for item in response.get("items", []):
            top_comment = item["snippet"]["topLevelComment"]
            snippet = top_comment["snippet"]

            comments.append({
                "comment_id": top_comment["id"],
                "video_id": video_id,
                "language": language,
                "comment_text": snippet.get("textOriginal", ""),
                "parent_comment_id": "",
            })

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

        time.sleep(0.1)

    return comments


def main():
    if not API_KEY:
        raise ValueError(
            "YOUTUBE_API_KEY is missing. Check the .env file."
        )

    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(
            f"Video inventory not found: {INVENTORY_FILE}"
        )

    inventory = pd.read_csv(INVENTORY_FILE, dtype=str).fillna("")

    all_comments = []

    youtube = build("youtube", "v3", developerKey=API_KEY)

    for _, video in inventory.iterrows():
        video_id = video["video_id"].strip()
        language = video["language"].strip()
        title = video["video_title"].strip()

        print(f"\nCollecting: {title}")
        print(f"Video ID: {video_id} | Language: {language}")

        try:
            comments = collect_comments(
                youtube,
                video_id,
                language,
                MAX_COMMENTS_PER_VIDEO,
            )

            all_comments.extend(comments)

            print(f"Collected: {len(comments)} comments")

        except HttpError as error:
            print(f"Could not collect comments for {video_id}: {error}")
            print("Continuing to the next video.")

    if not all_comments:
        print("No comments collected. Check the API errors above.")
        return

    new_df = pd.DataFrame(all_comments, columns=COLUMNS)

    # Remove duplicates within this collection.
    new_df = new_df.drop_duplicates(subset=["comment_id"])

    # Merge with existing data if the output file already exists.
    # Existing records are retained; comment IDs prevent duplication.
    if OUTPUT_FILE.exists():
        old_df = pd.read_csv(
            OUTPUT_FILE,
            dtype=str,
            keep_default_na=False,
        )
        combined_df = pd.concat(
            [old_df, new_df],
            ignore_index=True,
        )
        combined_df = combined_df.drop_duplicates(
            subset=["comment_id"],
            keep="first",
        )
    else:
        combined_df = new_df

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nCollection finished.")
    print(f"New unique comments: {len(new_df)}")
    print(f"Total unique comments in file: {len(combined_df)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nComments by language:")
    print(combined_df["language"].value_counts().to_string())


if __name__ == "__main__":
    main()
