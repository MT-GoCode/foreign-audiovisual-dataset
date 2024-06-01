from moviepy.editor import VideoFileClip
import moviepy.video.io.ffmpeg_tools as ffmpeg_tools
import pandas as pd
import random
import string
from .audio_functions import *
from .helper_objects import *
from .video_ML_functions import *
# from .FaceTracking.driver import *

class LocalFilter:
    def __init__(self, **kwargs):
        self.__dict__ = dict(kwargs)
        
        self.video_csv_pd = pd.read_csv(self.video_csv)
        self.clip_csv_pd = pd.read_csv(self.clip_csv)

    def fetch_remaining_videos(self):
        X = self.video_csv_pd[(self.video_csv_pd['PROCESSED?'] != 1) & 
                  (self.video_csv_pd['Language'] != 'en') & 
                  (self.video_csv_pd['Duration'] < 1200) & 
                  (self.video_csv_pd['Resolution'].str.len() >= 9)]['VIDEO ID'].tolist()
        return X[:min(len(X), self.limit_videos)]
    
    def execute(self):
        
        vid_ids_to_process = self.fetch_remaining_videos()

        print("videos to process: ")
        print(vid_ids_to_process)

        for VIDEO_ID in vid_ids_to_process:
            try:
                video_path = self.raw_folder + VIDEO_ID + ".mp4"
                print("ANALYZING VIDEO: " + video_path)

                test = Video_Custom(video_path)
                shots = test.detect_shots()

                # DETECT SHOTS THAT HAVE ONE VISIBLE FACE ONLY
                functional_shots = []
                for i, shot in enumerate(shots):
                    print("shot " + str(i+1) + "/" + str(len(shots)) + str(shot))
                    
                    if single_face_and_size_checker(shot, test, 3, self.min_face_size):
                        functional_shots.append(shot)
                        
                
                print("DETECTED TO HAVE ONE FACE: ")
                print(functional_shots)

                if len(functional_shots) == 0: continue

                wav = Audio_Custom.extract_audio(video_path)



                timestamps = vad_audio(wav)

                # APPLY VAD SMOOTHING
                smoothed_segments = smooth_voice_segments(timestamps, self.voice_detection_smoothing)
                print(smoothed_segments)

                if len(smoothed_segments) == 0: continue

                video = VideoFileClip(video_path)

                VAD_frames = voice_activity_to_frames(smoothed_segments, video.fps)

                # IDENTIFY SHOTS WITH PEOPLE SPEAKING, based on pyscenedetect + VAD

                good_speaking_shots = []

                for i, shot in enumerate(VAD_frames):
                    print("shot " + str(i+1) + "/" + str(len(VAD_frames)))
                    print("boundaries: ")
                    print(shot)
                    upper_lip_idx = 12
                    lower_lip_idx = 15
                    try:
                        if analyze_speaking(test, shot[0], shot[1], upper_lip_idx, lower_lip_idx, frame_step=5):
                            print("found!")
                            good_speaking_shots.append(shot);
                    except Exception as e:
                        print("SYSTEM: error while analyzing mouth movements.")
                        print(e)
                        continue;
                print("DETECTED TO HAVE SOMEONE SPEAKING")
                print(good_speaking_shots)

                # FINDING SHOTS WITH BOTH FACE SHOWING AND PEOPLE SPEAKING


                def find_common_segments(arr1, arr2):
                    events = [(t, 'start', 'A') for t, _ in arr1] + [(t, 'end', 'A') for _, t in arr1] + \
                            [(t, 'start', 'B') for t, _ in arr2] + [(t, 'end', 'B') for _, t in arr2]
                    events.sort(key=lambda x: (x[0], x[1] == 'end'))
                    
                    active_a = active_b = 0
                    last_start = None
                    common_segments = []
                    
                    for time, type, array in events:
                        if type == 'start':
                            if array == 'A':
                                active_a += 1
                            else:
                                active_b += 1
                            if active_a > 0 and active_b > 0 and last_start is None:
                                last_start = time
                        else:
                            if active_a > 0 and active_b > 0 and last_start is not None:
                                common_segments.append((last_start, time))
                                last_start = None
                            if array == 'A':
                                active_a -= 1
                            else:
                                active_b -= 1
                    
                    return common_segments


                overlap = find_common_segments(functional_shots, good_speaking_shots)

                print("overlapping frames:")
                print(overlap)

                for i, (start, end) in enumerate(overlap):

                    start_time = start / video.fps
                    end_time = end / video.fps

                    print("time length: ")
                    print(end_time - start_time)
                    print("min requirement: " + str(self.min_length))
                    if (end_time - start_time < self.min_length):
                        continue

                    CLIP_ID = ''.join(random.choices(string.ascii_letters + string.digits, k=15))

                    print("SAVING THESE TIMESTAMPS")
                    print(start_time)
                    print(end_time)

                    # CLIPPING & SAVING
                    output_filename = self.clip_folder + f'{CLIP_ID}.mp4'  # Naming clips sequentially
                    clip = video.subclip(start_time, end_time)
                    clip.write_videofile(output_filename, codec="libx264")
                    
                    # RECORDING DATA
                    record = pd.Series([CLIP_ID, VIDEO_ID, start_time, end_time], index = ["CLIP ID", "VIDEO ID", "START TIME", "END TIME"])
                    self.update_clip_csv(record)
                
                print("SYSTEM: done clipping for video " + str(VIDEO_ID))
                self.mark_video_processed(VIDEO_ID)

                video.close()

            except Exception as e:
                print("SYSTEM: something went wrong for video " + VIDEO_ID)
                print(e)
                continue;

    def mark_video_processed(self, VIDEO_ID):
        self.video_csv_pd.loc[self.video_csv_pd['VIDEO ID'] == VIDEO_ID, 'PROCESSED?'] = 1

        # self.query_csv is the PATH
        self.video_csv_pd.to_csv(self.video_csv, index=False) 

    def update_clip_csv(self, row):
        print("row to save: ")
        print(row)

        self.clip_csv_pd.loc[len(self.clip_csv_pd)] = row  # adding a row

        # self.clip_csv is the PATH
        self.clip_csv_pd.to_csv(self.clip_csv, index=False)



