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

The pipeline is customizable and incrementing: It can be stopped, started any time without damage to successfully-collected data. Different parts of the pipeline can be turned off and on depending on your needs.

```yaml
Collection_Pipeline:

    # turn off different parts of pipeline by setting enabled: False
    download:
        enabled: False
        query_csv: CSV/video_queries.csv # should be a link to a csv file containing queries to use for youtube
        raw_folder: DATA/raw_videos/ # this is where  videos will be stored after download. FORMATTING: Make sure there is a slash at the end.
        video_csv: CSV/raw_videos.csv # this is where where downloaded videos will be recorded. FORMATTING: Make sure there is a slash at the end.
        results_per_query: 10 # how many videos to download per query
        limit_query: 1000 # how may queries you'd like to process at maximum, before the download phase completes.
        limit_videos: 500 # a hard limit of how many videos to download in one running of the download phase.

        exclude_english: True # sometimes, youtube gives language tags. If it says "en," should we exclude that video?

    filter:
        enabled: True
        download_clips: True
        clip_folder: DATA/video_clips/
        # This is where video clips will be stored after filtering a video down to key scenes 
        # This is ALSO where the "analysis" stage will source videos

        clip_csv: CSV/video_clips.csv # this is a CSV file with one record per clip 
        min_face_size: 100 # minimum size in pixels for a face to be included
        min_length: 5 # minimum length of a segment, in seconds
        limit_videos: 500 # limit how many videos we'd like to filter and clip for now
        voice_detection_smoothing: 0.5 # in seconds, the pause which would cause two different speech segments to be considered part of the same same.

    analysis:
    # 3 forms of clip-by-clip analysis can be done: 
        # crop - crop a video clip down to the area around the face
        # attribution - analyse a clip for gender, ethnicity, facial hair
        # pose estimation - frame-by-frame pose estimation
        # Depending on what is set to true, the code will run just that part of analysis.
        enabled: False

        limit_videos: 1000 # how many clips to analyze at a time

        feature_csv: STATISTICS/features.csv
        # where to save attribute and "area of interest/crop" features.
        # See "feature.csv format" for expected format of this csv.

        crop: True
        download_crops: False # crops will be downloaded to the same folder as clip_folder, titled <clip>_cropped.mp4"
        
        attribute: True # whether to run the attribution model

        pose_estimation: True # whether to run pose estimation on every frame.
        pose_folder: STATISTICS/pose/
            # where to dump pose data - each clip will produce a file "<clip name>_pose.csv"

    post_processing:
        enabled: False

        plot: True # whether to generate some plots for attribution data. If you just want to plot, turn all the other parts of the pipeline to False
        plot_folder: STATISTICS/plots/ # where to put image plots


```
