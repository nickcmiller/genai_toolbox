import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import feedparser
import urllib
import requests

import logging
import traceback
import json
from typing import List

def parse_feed(
    feed_url: str
) -> feedparser.FeedParserDict:
    """
    Parses a podcast feed and returns the feed object.

    Arguments:
        feed_url: str - The URL of the podcast feed.

    Returns:
        feedparser.FeedParserDict - The feed object.
    """
    try:
        feed = feedparser.parse(feed_url)
        if feed is None:
            raise ValueError("Feed is None")
        return feed
    except Exception as e:
        logging.error(f"Failed to parse feed: {e}")
        traceback.print_exc()
        return None

def extract_entry_metadata(
    entry: dict
) -> dict:
    """
    Extracts metadata from an entry in a podcast feed.

    Arguments:
        entry: dict - The entry to extract metadata from.

    Returns:
        dict - The extracted metadata.
    """
    entry_id = entry.id
    title = entry.title
    published = entry.published
    summary = entry.summary
    url = next((link['href'] for link in entry.links if link.rel == 'enclosure'), None)
    
    return {
        "entry_id": entry_id,
        "title": title,
        "published": published,
        "summary": summary,
        "url": url
    }

def extract_metadata_from_feed(
    feed: feedparser.FeedParserDict
) -> dict:
    """
    Extracts metadata from a podcast feed.


    """
    entries = []
    
    for entry in feed.entries:
        entry_metadata = extract_entry_metadata(entry)
        entry_metadata["feed_summary"] = feed.feed.summary
        entries.append(entry_metadata)
    
    return entries

def return_entries_from_feed(
    feed_url: str
) -> List[dict]:
    """
        Returns a list of entries from a podcast feed.

        Arguments:
            feed_url: str - The URL of the podcast feed.

        Returns:
            List[dict] - A list of entries from the podcast feed.

        Example:
            >>> result = return_entries_from_feed(
                feed_url="https://feeds.simplecast.com/3hnxp7yk"
            )
            >>> print(result)
            [{'entry_id': 'd8029cde-4677-4ac9-bdbc-2f05fba1c1c5', 
            'title': '#351 The Founder of Rolex: Hans Wilsdorf', 
            'published': 'Tue, 4 Jun 2024 16:05:59 +0000', 
            'summary': 'What I learned from reading about Hans Wilsdorf and the founding of Rolex.', 
            'url': 'https://www.founderspodcast.com/', 
            'feed_summary': 'Learn from history\'s greatest entrepreneurs...'}, 
            {...}, {...}]
    """
    feed = parse_feed(feed_url)
    return extract_metadata_from_feed(feed)

def download_podcast_audio(
    audio_url: str, 
    title: str, 
    file_path: str=None
) -> str:
    """
        Downloads a podcast audio file from a URL and saves it to a file.

        Arguments:
            audio_url: str - The URL of the podcast audio file.
            title: str - The title of the podcast episode.
            file_path: str - The path to save the podcast audio file to.

        Returns:
            str - The path to the saved podcast audio file.

        Example:
            >>> result = download_podcast_audio(
                audio_url="https://www.example.com/podcast.mp3",
                title="Podcast Episode Title",
                file_path="/path/to/save/podcast.mp3"
            )
            >>> print(result)
            "/path/to/save/podcast.mp3"
    """
    if file_path is None:
        file_path = os.getcwd() + "/"
    
    safe_title = ''.join(char for char in title if char.isalnum() or char in " -_")
    title_with_underscores = safe_title.replace(" ", "_")
    file_name = os.path.join(file_path, title_with_underscores + ".mp3")
    
    response = requests.get(audio_url)
    
    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(response.content)
        logging.info(f"File downloaded: {title}")
    else:
        logging.error(f"Failed to download the file: {title}")

    return file_name

if __name__ == "__main__":
    print(json.dumps(return_entries_from_feed("https://feeds.megaphone.fm/ATHLLC5883700320")[0], indent=4))