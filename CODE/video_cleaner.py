import pandas as pd
import os

# Load the video IDs from the CSV
csv_path = r"C:\Users\tminh\Downloads\DatasetGenerator\CSV\raw_videos.csv"
video_csv_pd = pd.read_csv(csv_path)
video_ids = set(video_csv_pd['VIDEO ID'].tolist())

# Directory containing the video files
data_dir = r"C:\Users\tminh\Downloads\DatasetGenerator\DATA\raw_videos"

# Get list of all video files
all_videos = [f for f in os.listdir(data_dir) if f.endswith('.mp4')]

# Filter out videos that should not be deleted
videos_to_keep = [video for video in all_videos if video[:-4] in video_ids]
print("Videos to NOT be deleted:", videos_to_keep)

# Delete unmatched video files
for video in all_videos:
    if video not in videos_to_keep:
        os.remove(os.path.join(data_dir, video))
        print(f"Deleted {video}")
