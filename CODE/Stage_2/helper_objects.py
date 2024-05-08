import cv2
import os
from matplotlib import pyplot as plt
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector


class Video_Custom:
    def __init__(self, filepath):
        self.filepath = filepath

    # Function to find scenes.
    def detect_shots(self):
      # Create a video manager object for the video.
      video_manager = VideoManager([self.filepath])
      scene_manager = SceneManager()

      # Add ContentDetector algorithm (constructor takes optional threshold argument).
      scene_manager.add_detector(ContentDetector())

      # Start the video manager and perform scene detection.
      video_manager.start()
      scene_manager.detect_scenes(frame_source=video_manager)

      # Once detection is done, we retrieve the scene list
      scene_list = scene_manager.get_scene_list()

      # We finalize the video manager
      video_manager.release()

      # Convert the scene list to a list of tuples (start_frame, end_frame)
      shots = [(start.get_frames(), end.get_frames()) for start, end in scene_list]
      return shots


    def get_frame(self, num):
        cap = cv2.VideoCapture(self.filepath)

        cap.set(cv2.CAP_PROP_POS_FRAMES, num)

        # Read the frame
        ret, frame = cap.read()
        cap.release()

        return frame

    def display_frame(self, num):
        frame_to_display = self.get_frame(num)

        # Check if the frame was successfully retrieved
        if frame_to_display is None or frame_to_display.size == 0:
            print(f"Frame {num} is empty or could not be loaded.")
        else:
            frame_to_display_rgb = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)

            plt.imshow(frame_to_display_rgb)
            plt.title(f'Frame {num}')
            plt.axis('off')
            plt.show()

from moviepy.editor import *
import matplotlib.pyplot as plt

class Audio_Custom:

    def extract_audio(vid_filepath):
        video = VideoFileClip(vid_filepath)
        audio = video.audio
        save_path = vid_filepath.rsplit('.', 1)[0] + '.wav'
        audio.write_audiofile(save_path)
        return save_path

    def __init__(self, filepath):
        self.filepath = filepath