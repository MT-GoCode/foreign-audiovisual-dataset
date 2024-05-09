# foreign-audiovisual-dataset

## Usage

### Setup

```python
# python version >= 3.7

python -m venv audiovisual
source audiovisual/bin/activate

pip install requirements.txt
```

### Full pipeline; Youtube to Analyzed Video Clips

```python
python main.py --config config.yaml
```

### Configuration File (YAML)

```yaml
infrastructure: "local"

Collection_Pipeline:

    # turn off different parts of pipeline by setting enabled: False
    download:
        enabled: False
        query_csv: CSV/video_queries.csv # contains queries to use
        raw_folder: DATA/raw_videos/ # MAKE SURE THERE IS / at the end. If this is relative path, do not include / at the beginning. where videos will be stored after download
        video_csv: CSV/raw_videos.csv # where downloaded videos will be recorded
        results_per_query: 5
        limit_query: 2 # how may queries you'd like to process, then stop
        limit_videos: 50 # a hard limit of how many videos to download in one running

    clip:
        enabled: True
        download_clips: True
        clip_folder: DATA/video_clips/
        # This is where the 
        # This is ALSO where the "analysis" stage will source videos

        clip_csv: CSV/video_clips.csv # where individual clips will be tracked
        min_face_size: 100 # minimum size in pixels for a face to be included
        skip_crop: False # crop the video or not
        min_length: 5 # minimum length of a segment, in seconds
        limit_videos: 50 # limit how many videos we'd like to filter and clip for now
        voice_detection_smoothing: 0.5 # in seconds, the pause which would cause two different speech segments to be considered the same.

    analysis:
    # 3 forms of clip-by-clip analysis can be done: 
        # crop - crop a video clip down to the area around the face
        # attribution - analyse a clip for gender, ethnicity, facial hair
        # pose estimation - frame-by-frame pose estimation
        # Depending on what is set to true, the code will run just that part of analysis.

        limit_videos: 50 # how many clips to analyze at a time

        crop: True
        download_crops: True # crops will be downloaded to the same folder as clip_folder
        download_replace: True 
            # True: crop videos destructively 
            # False: save a copy called "<clip>_cropped.mp4"
        
        attribute: True
        attribute_csv: STATISTICS/attribute.csv
            # where to save statistics of each video.
            # See "attribute.csv format" for expected format of this filepath.

        pose_estimation: True
        pose_folder: STATISTICS/pose/
            # where to dump pose data - each clip will produce a file "<clip name>_pose.csv"
   
    encord_upload: 
        enabled: True
        source_folder: CSV/video_clips.csv

```
