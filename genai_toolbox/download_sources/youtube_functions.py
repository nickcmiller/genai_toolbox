import yt_dlp
# from pydub import AudioSegment
import googleapiclient.discovery

import os
import re
import logging
import traceback
from dotenv import load_dotenv

# Get the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


dotenv_path = os.path.join(parent_dir, '.env')
load_dotenv(dotenv_path, override=True)
# load_dotenv()

logging.basicConfig(level=logging.INFO)

from genai_toolbox.helper_functions.datetime_helpers import get_date_with_timezone
from genai_toolbox.text_prompting.model_calls import openai_text_response, groq_text_response

def yt_dlp_download(
    yt_url_or_id: str, 
    output_path: str = None
) -> str:
    """
    Downloads the audio track from a specified YouTube video URL or ID using the yt-dlp library, then converts it to an MP3 format file.

    Args:
        yt_url_or_id (str): The URL or ID of the YouTube video from which audio will be downloaded. This should be a valid YouTube video URL or ID.

    Returns:
        str: The absolute file path of the downloaded and converted MP3 file. This path includes the filename which is derived from the original video title.

    Raises:
        yt_dlp.utils.DownloadError: If there is an issue with downloading the video's audio due to reasons such as video unavailability or restrictions.
        Exception: For handling unexpected errors during the download and conversion process.
    """
    if not is_valid_youtube_url_or_id(yt_url_or_id):
        raise ValueError(f"Invalid YouTube URL or ID: {yt_url_or_id}")

    if not yt_url_or_id.startswith(('http://', 'https://')):
        yt_url_or_id = f"https://youtu.be/{yt_url_or_id}"

    if output_path is None:
        output_path = os.getcwd()

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
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

def get_channel_and_video_metadata(
    api_key: str, 
    channel_id: str, 
    start_date: str = None, 
    end_date: str = None
) -> dict:
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    channel_metadata = _get_channel_metadata(youtube, channel_id)
    video_metadata = _get_video_metadata(youtube, channel_metadata, start_date, end_date)

    return video_metadata

def _get_channel_metadata(
    youtube: googleapiclient.discovery.Resource, 
    channel_id: str
) -> dict:
    channel_response = youtube.channels().list(
        part="snippet,brandingSettings,contentDetails",
        id=channel_id
    ).execute()
    channel_info = channel_response["items"][0]

    return {
        "channel_id": channel_id,
        "channel_title": channel_info["snippet"]["title"],
        "channel_description": channel_info["snippet"]["description"],
        "channel_keywords": channel_info["brandingSettings"]["channel"].get("keywords", ""),
        "uploadsPlaylistId": channel_info["contentDetails"]["relatedPlaylists"]["uploads"]
    }

def _get_video_metadata(
    youtube: googleapiclient.discovery.Resource, 
    channel_metadata: dict, 
    start_date: str = None, 
    end_date: str = None
) -> list:
    video_metadata = []
    next_page_token = None

    # Convert date strings to datetime objects
    published_after = get_date_with_timezone(start_date) if start_date else None
    published_before = get_date_with_timezone(end_date) if end_date else None

    while True:
        request_params = {
            "part": "contentDetails,snippet,status",
            "playlistId": channel_metadata["uploadsPlaylistId"],
            "maxResults": 50,
            "pageToken": next_page_token
        }

        response = youtube.playlistItems().list(**request_params).execute()

        for item in response["items"]:
            published_at = get_date_with_timezone(item["snippet"]["publishedAt"])
            if (not published_after or published_at >= published_after) and (not published_before or published_at <= published_before):
                video_info = {
                    "video_id": item["contentDetails"]["videoId"],
                    "video_title": item["snippet"]["title"],
                    "video_description": item["snippet"]["description"],
                    "published_at": item["snippet"]["publishedAt"],
                    **{k: v for k, v in channel_metadata.items() if k != "uploadsPlaylistId"}
                }
                video_metadata.append(video_info)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return video_metadata

def generate_episode_summary(
    video_title: str,
    video_description: str,
    channel_keywords: str,
    channel_title: str,
    channel_description: str
) -> str:
    summary_prompt = f"""
        {'-'*30}\n 
        Description of the YouTube channel:\n 
        {'-'*30}\n 
        {channel_title} \n
        {channel_description} \n
        Here are keywords for the channel: {channel_keywords}
        \n
        {'-'*30}\n\n
        Description of this specific YouTube video:\n
        {'-'*30}\n 
        {video_title} \n
        {video_description} \n
        {'-'*30}\n\n
        Describe the hosts and the guests expected in this specific episode.
    """
    print(summary_prompt)
    
    try:
        response = openai_text_response(
            model_choice="4o-mini",
            prompt=summary_prompt
        )        
    except Exception as e:
        logging.error(f"OpenAI model call failed: {e}")
        try:
            response = groq_text_response(
                model_choice="llama3-70b",
                prompt=summary_prompt
            )
        except Exception as e:
            logging.error(f"Groq model call failed: {e}")
            response = "Failed to generate summary due to API errors."
        
    return response

# Example usage
import json
import os

api_key = os.getenv("YOUTUBE_API_KEY")
channel_id = "UCyaN6mg5u8Cjy2ZI4ikWaug"
start_date = "2024-06-28" 
end_date = "2024-07-01" 

video_metadata = get_channel_and_video_metadata(api_key, channel_id, start_date, end_date)

for video in video_metadata:
    print(json.dumps(video, indent=4))
    summary = generate_episode_summary(
        video_title=video["video_title"],
        video_description=video["video_description"],
        channel_keywords=video["channel_keywords"],
        channel_title=video["channel_title"],
        channel_description=video["channel_description"]
    )
    video["summary"] = summary