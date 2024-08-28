import yt_dlp
import isodate
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError

from datetime import datetime
import logging
import os
import re
from typing import List, Dict, Optional
import traceback

from genai_toolbox.helper_functions.datetime_helpers import get_date_with_timezone
from genai_toolbox.text_prompting.model_calls import openai_text_response, groq_text_response, anthropic_text_response

logging.basicConfig(level=logging.INFO)

# Constants
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
MAX_RESULTS_PER_PAGE = 50

def yt_dlp_download(
    yt_url_or_id: str, 
    output_dir: str = None
) -> str:
    """
        Downloads the audio track from a specified YouTube video URL or ID using the yt-dlp library, then converts it to an MP3 format file.

        Args:
            yt_url_or_id (str): The URL or ID of the YouTube video from which audio will be downloaded. This should be a valid YouTube video URL or ID.
            output_dir (str, optional): The directory where the downloaded audio file will be saved. Defaults to the current working directory.

        Returns:
            str: The absolute file path of the downloaded and converted MP3 file. This path includes the filename which is derived from the original video title.

        Raises:
            ValueError: If the input YouTube URL or ID is invalid.
            yt_dlp.utils.DownloadError: If there is an issue with downloading the video's audio due to reasons such as video unavailability or restrictions.
            Exception: For handling unexpected errors during the download and conversion process.
    """
    if not is_valid_youtube_url_or_id(yt_url_or_id):
        raise ValueError(f"Invalid YouTube URL or ID: {yt_url_or_id}")

    if not yt_url_or_id.startswith(('http://', 'https://')):
        yt_url_or_id = f"https://youtu.be/{yt_url_or_id}"

    if output_dir is None:
        output_dir = os.getcwd()

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(yt_url_or_id, download=True)
            file_name = ydl.prepare_filename(result)
            mp3_file_path = file_name.rsplit('.', 1)[0] + '.mp3'
            logging.info(f"yt_dlp_download saved YouTube video to file path: {mp3_file_path}")
            return mp3_file_path
    except yt_dlp.utils.DownloadError as e:
        logging.error(f"yt_dlp_download failed to download audio from URL {yt_url_or_id}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred with yt_dlp_download: {e}")
        logging.error(traceback.format_exc())
        raise

def is_valid_youtube_url_or_id(
    input_str: str
) -> bool:
    # Regular expression for YouTube URLs
    youtube_url_pattern = re.compile(
        r'^(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$'
    )
    
    # Regular expression for YouTube video IDs
    youtube_id_pattern = re.compile(
        r'^[A-Za-z0-9-_]{11}$'
    )
    
    # Check if input matches YouTube URL pattern
    if youtube_url_pattern.match(input_str):
        return True
    
    # Check if input matches YouTube video ID pattern
    if youtube_id_pattern.match(input_str):
        return True
    
    return False

def retrieve_youtube_channel_and_video_metadata_by_date(
    api_key: str, 
    channel_id: str, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None,
    short_duration: int = 10*60
) -> List[Dict]:
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=api_key)

    try:
        channel_metadata = _get_channel_metadata(youtube, channel_id)
        video_metadata = _get_video_metadata(youtube, channel_metadata, start_date, end_date, short_duration)
        return video_metadata
    except HttpError as e:
        logging.error(f"An HTTP error occurred: {e}")
        raise

def _get_channel_metadata(
    youtube: Resource, 
    channel_id: str
) -> Dict:
    try:
        channel_response = youtube.channels().list(
            part="snippet,brandingSettings,contentDetails",
            id=channel_id
        ).execute()
        
        if not channel_response.get("items"):
            raise ValueError(f"No channel found for ID: {channel_id}")
        
        channel_info = channel_response["items"][0]
        return {
            "channel_id": channel_id,
            "feed_title": channel_info["snippet"]["title"],
            "feed_description": channel_info["snippet"]["description"],
            "feed_keywords": channel_info["brandingSettings"]["channel"].get("keywords", ""),
            "uploadsPlaylistId": channel_info["contentDetails"]["relatedPlaylists"]["uploads"]
        }
    except HttpError as e:
        logging.error(f"Error fetching channel metadata: {e}")
        raise

def _get_video_metadata(
    youtube: Resource, 
    channel_metadata: Dict, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None,
    short_duration: int = 10*60
) -> List[Dict]:
    video_metadata = []
    published_after = get_date_with_timezone(start_date) if start_date else None
    published_before = get_date_with_timezone(end_date) if end_date else None

    request_params = {
        "part": "contentDetails,snippet,status",
        "playlistId": channel_metadata["uploadsPlaylistId"],
        "maxResults": MAX_RESULTS_PER_PAGE
    }

    try:
        while True:
            response = youtube.playlistItems().list(**request_params).execute()
            
            for item in response["items"]:
                video_id = item["contentDetails"]["videoId"]
                published_at = get_date_with_timezone(item["snippet"]["publishedAt"])
                if _is_within_date_range(published_at, published_after, published_before):
                    # Get additional video details to check duration
                    video_response = youtube.videos().list(
                        part="contentDetails",
                        id=video_id
                    ).execute()
                    
                    if video_response["items"]:
                        duration = video_response["items"][0]["contentDetails"]["duration"]
                        if not _is_short_video(duration, short_duration):
                            video_info = {
                                "video_id": video_id,
                                "title": item["snippet"]["title"],
                                "description": item["snippet"]["description"],
                                "published": item["snippet"]["publishedAt"],
                                **{k: v for k, v in channel_metadata.items() if k != "uploadsPlaylistId"}
                            }
                            video_metadata.append(video_info)
            
            if "nextPageToken" not in response:
                break
            request_params["pageToken"] = response["nextPageToken"]
    
    except HttpError as e:
        logging.error(f"Error fetching video metadata: {e}")
        raise

    return video_metadata

def _is_short_video(
    duration: str,
    short_duration: int = 10*60
) -> bool:
    """
    Check if a video is a Short based on its duration.
    YouTube Shorts are typically 60 seconds or less.
    """
    try:
        duration_timedelta = isodate.parse_duration(duration)
        total_seconds = duration_timedelta.total_seconds()
        return total_seconds <= short_duration
    except isodate.ISO8601Error:
        logging.warning(f"Invalid duration format: {duration}")
        return False

def _is_within_date_range(
    date: datetime, 
    start: Optional[datetime], 
    end: Optional[datetime]
) -> bool:
    return (not start or date >= start) and (not end or date <= end)

def generate_episode_summary(
    title: str,
    description: str,
    feed_keywords: str,
    feed_title: str,
    feed_description: str,
) -> str:
    host_guest_prompt = f"""
        Using the information below, concisely describe the hosts and the guests expected in this specific episode. Use bullet points to list the hosts and guests.

        {'-'*30}\n 
        Description of the YouTube channel:\n 
        {'-'*30}\n 
        {feed_title} \n
        {feed_description} \n
        Here are keywords for the channel: {feed_keywords}
        \n
        {'-'*30}\n\n
        Description of this specific YouTube video:\n
        {'-'*30}\n 
        {title} \n
        {description} \n
        {'-'*30}\n\n
    """
    host_guest_system_prompt = "You are a helpful assistant that helps me identify the hosts and guests in a YouTube video."

    host_guest_summary = openai_text_response(
        model_choice="4o-mini",
        prompt=host_guest_prompt,
        system_instructions=host_guest_system_prompt
    )

    return host_guest_summary
    



if __name__ == "__main__":
    # Example usage
    import json
    import os

    # Get the parent directory
    from dotenv import load_dotenv
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv_path = os.path.join(parent_dir, '.env')
    load_dotenv(dotenv_path, override=True)

    api_key = os.getenv("YOUTUBE_API_KEY")
    channel_id = "UCyaN6mg5u8Cjy2ZI4ikWaug"
    start_date = "2024-02-01" 
    end_date = "2024-03-01" 

    video_metadata = retrieve_youtube_channel_and_video_metadata_by_date(api_key, channel_id, start_date, end_date)

    print(f"Number of videos found: {len(video_metadata)}")
    for video in video_metadata:
        # print(json.dumps(video, indent=4))
        formatted_date = datetime.fromisoformat(video['published'][:-1]).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{formatted_date} - {video['title']}")