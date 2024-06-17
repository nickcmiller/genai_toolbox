import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import feedparser
import urllib
from dateutil import parser
import pytz
from datetime import datetime

import requests
import logging
import traceback
import json
from typing import List, Optional

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

        Example:
            >>> result = extract_entry_metadata(parse_feed("https://feeds.simplecast.com/3hnxp7yk")[0])
            >>> print(result)
            {'entry_id': 'd8029cde-4677-4ac9-bdbc-2f05fba1c1c5', 
            'title': '#351 The Founder of Rolex: Hans Wilsdorf', 
            'published': 'Tue, 4 Jun 2024 16:05:59 +0000', 
            'summary': 'What I learned from reading about Hans Wilsdorf and the founding of Rolex.', 
            'url': 'https://www.founderspodcast.com/', 
            'feed_summary': 'Learn from history\'s greatest entrepreneurs...'}
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
) -> List[dict]:
    """
        Extracts metadata from a podcast feed.

        Arguments:
            feed: feedparser.FeedParserDict - The feed object.

        Returns:
            List[dict] - A list of entries from the podcast feed.

        Example:
            >>> result = extract_metadata_from_feed(parse_feed("https://feeds.simplecast.com/3hnxp7yk"))
            >>> print(result)
            {'entry_id': 'd8029cde-4677-4ac9-bdbc-2f05fba1c1c5', 
            'title': '#351 The Founder of Rolex: Hans Wilsdorf', 
            'published': 'Tue, 4 Jun 2024 16:05:59 +0000', 
            'summary': 'What I learned from reading about Hans Wilsdorf and the founding of Rolex.', 
            'url': 'https://www.founderspodcast.com/', 
            'feed_summary': 'Learn from history\'s greatest entrepreneurs...'}, 
            {...}, {...}]
    """
    entries = []
    
    for entry in feed.entries:
        entry_metadata = extract_entry_metadata(entry)
        entry_metadata["feed_summary"] = feed.feed.summary
        entries.append(entry_metadata)
    
    return entries

def return_all_entries_from_feed(
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

def get_date_with_timezone(
    date_str: str, 
    timezone_str: str = 'UTC'
) -> datetime:
    """
        Parses a date string and returns a datetime object localized to the specified timezone.

        Args:
            date_str (str): The date string to parse.
            timezone_str (str): The timezone to localize the date to. Defaults to 'UTC'.

        Returns:
            datetime: A datetime object localized to the specified timezone.

        Raises:
            ValueError: If the date string is invalid or the timezone string is not recognized.

        Examples:
            >>> get_date_with_timezone("2024-06-01")
            datetime.datetime(2024, 6, 1, 0, 0, tzinfo=<UTC>)
            >>> get_date_with_timezone("June 1st, 2024", "America/New_York")
            datetime.datetime(2024, 6, 1, 0, 0, tzinfo=<DstTzInfo 'America/New_York' EST-1 day, 19:00:00 STD>)
    """
    try:
        naive_date = parser.parse(date_str)
    except ValueError as e:
        raise ValueError(f"Invalid date string: {date_str}. Error: {e}")

    try:
        timezone = pytz.timezone(timezone_str)
    except pytz.UnknownTimeZoneError as e:
        raise ValueError(f"Unknown timezone: {timezone_str}. Error: {e}")

    if naive_date.tzinfo is None:
        localized_date = timezone.localize(naive_date)
    else:
        localized_date = naive_date.astimezone(timezone)

    return localized_date

def return_entries_by_date(
    feed_url: str,
    start_date_str: str,
    end_date_str: Optional[str] = None
) -> List[dict]:
    """
        Retrieves podcast entries from a specified feed URL that are published within a given date range.

        Args:
            feed_url (str): The URL of the podcast feed.
            start_date_str (str): The start date as a string in YYYY-MM-DD format. Defaults to "1900-01-01".
            end_date_str (Optional[str]): The end date as a string in YYYY-MM-DD format. Defaults to today's date.

        Returns:
            List[dict]: A list of podcast entries that fall within the specified date range.

        Raises:
            ValueError: If the start_date_str or end_date_str cannot be parsed into a date.
    """
    # Set default end date to today if not provided
    if end_date_str is None:
        end_date_str = datetime.now().strftime("%Y-%m-%d")

    try:
        start_date = get_date_with_timezone(start_date_str)
        end_date = get_date_with_timezone(end_date_str)
    except ValueError as e:
        raise ValueError(f"Error parsing date strings: {e}")

    entries = return_all_entries_from_feed(feed_url)
    filtered_entries = [
        entry for entry in entries
        if start_date <= get_date_with_timezone(entry['published']) <= end_date
    ]

    return filtered_entries

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