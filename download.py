import pandas as pd
from youtubesearchpython import VideosSearch
import json

def get_youtube_data(query, results_limit=5):
    video_search = VideosSearch(query, limit=results_limit)
    results = video_search.result()
    return [x['id'] for x in results['result'] ]

    # returns a bunch of ID's

import yt_dlp
import pandas as pd

def get_video_info(url):

    ydl_opts = {
        'skip_download': True,  # Do not download the video
        'force_generic_extractor': True,  # This might be necessary if yt-dlp is unable to download video info directly
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Fetch video information
        info_dict = ydl.extract_info(url, download=False)
        # Since '--print-json' is not directly available, we work with the info_dict returned
        print(info_dict)
        return info_dict

def produce_video_record(url):

    # try except to account for video being copyrighted
    try:
        info = get_video_info(url)
        record = pd.Series([info.get('id'), info.get('title', 'N/A'), url, f"{info.get('width', 'N/A')}x{info.get('height', 'N/A')}", info.get('duration', 'N/A'), info.get('language', 'N/A')],
            index = ['VIDEO ID', 'Name', 'Video Link', 'Resolution', 'Duration', 'Language'])
        return record
    except Exception as e:
        print("Could not get video url: ", e)
        return pd.Series(['N/A', 'N/A', url, 'N/A', 'N/A', 'N/A'], index=['VIDEO ID', 'Name', 'Video Link', 'Resolution', 'Duration', 'Language'])



def download_video(video_url, desired_name, output_path):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': VIDEO_OUTPUT_PATH + desired_name + '.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_url])
            print("Video downloaded successfully.")
        except Exception as e:
            print("An error occurred while downloading the video: ", e)
