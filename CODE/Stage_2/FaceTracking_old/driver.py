import pandas as pd
import json
import os
from tracking import VideoProcessor
from moviepy.editor import VideoFileClip
from PoseEstimation import *

class ClipProcessor:

    def convert_keys_to_strings(d):
        if isinstance(d, dict):
            return {str(k): ClipProcessor.convert_keys_to_strings(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [ClipProcessor.convert_keys_to_strings(v) for v in d]
        else:
            return d

    def process_csv_and_videos(video_path, clip_start, clip_end):


            processor = VideoProcessor()
            grouped_bbox, text_attributes = processor.proccess_video(video_path)

            results = {
                'start': clip_start,
                'end': clip_end,
                'grouped_bbox': grouped_bbox,
                'text_attributes': text_attributes
            }
            # results = LocalCropper.convert_keys_to_strings(results)

            print(results)
            
            return results

    def process(video_path, clip_start, clip_end):
        results = ClipProcessor.process_csv_and_videos(
            video_path = video_path,
            clip_start = clip_start,
            clip_end = clip_end)
        
        print("OLD FACETRACKING RESULTS: ")
        print(result)
        
        _ = ClipProcessor.find_furthest_points(results['grouped_bbox'])
        box_for_crop = next(iter(_.values()))

        print("finished processing "+ video_path)

        return {"crop_box": box_for_crop, "text_attributes": results['text_attributes']}

    # def pose_estimate(clip_path):
    #     return run(clip_path)

    def crop_and_trim_video(video_path, output_name, bbox, clip_start, clip_end, inplace=False):

        with VideoFileClip(video_path) as clip:
            cropped = clip.crop(x1=bbox[0], y1=bbox[1], width=bbox[2], height=bbox[3]).subclip(clip_start, clip_end)
            try:
                cropped.write_videofile(output_name)
            except Exception as e:
                print(f"An error occurred while writing the video file: {e}")
        
        
    def find_furthest_points(bboxes_dict):
        furthest_points = {}
        for cluster_id, bboxes in bboxes_dict.items():
            min_x = min_y = float('inf')
            max_x = max_y = -float('inf')

            for bbox in bboxes.values():
                x, y, width, height = bbox
                # Update min and max x
                min_x = min(min_x, x)
                max_x = max(max_x, x + width)
                # Update min and max y
                min_y = min(min_y, y)
                max_y = max(max_y, y + height)

        # Return the bounding box in the format (x, y, width, height)
        return (min_x, min_y, max_x - min_x, max_y - min_y)
            #     min_x = min(min_x, x-(width*.15))
            #      min_y = min(min_y, y-(height*.15))
            #     max_x = max(max_x, (x + width)+(width*.15))
            #     max_y = max(max_y, (y + height)+(height*.15))
            # furthest_points[cluster_id] = (min_x, min_y, max_x-min_x, max_y-min_y)

        # return furthest_points


# LocalCropper.execute(video_path = r"C:\Users\tminh\Downloads\DatasetGenerator\DATA\raw_videos\EpkalXgckaw.mp4",
#             clip_start = 0.16683333333333333,
#             clip_end = 12.178833333333332)

ClipProcessor.process(video_path = r"C:\Users\tminh\Downloads\DatasetGenerator\DATA\video_clips\Fmrcd8ebQMmjjWJ.mp4", clip_start = 0, clip_end = 10)


# print(ClipProcessor.pose_estimate(r"C:\Users\tminh\Downloads\DatasetGenerator\DATA\video_clips\test_cropped.mp4"))