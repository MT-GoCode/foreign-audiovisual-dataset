from .FaceTracking.driver import *
from moviepy.editor import VideoFileClip

class LocalAnalyzer:

    def __init__(self, **kwargs):
        self.__dict__ = dict(kwargs)

        self.feature_csv_pd = pd.read_csv(self.feature_csv)
        self.clip_csv_pd = pd.read_csv(self.clip_csv)
        self.video_csv_pd = pd.read_csv(self.video_csv)

    def fetch_remaining_clips(self):
        X = self.clip_csv_pd[self.clip_csv_pd['PROCESSED?'] != 1]['CLIP ID'].tolist() # .tolist()
        return X[:min(len(X), self.limit_videos)]
    
    def analyze_clip(self, video_path):
        clip = VideoFileClip(video_path)

        DATA = {}

        if self.crop or self.attribute:
            _ = ClipProcessor.process_crop_and_attr(video_path = video_path, clip_start = 0, clip_end = clip.duration)
            DATA['crop_box'] = _['crop_box']
            DATA['text_attributes'] = _['text_attributes']

            if self.download_crops:
                ClipProcessor.crop_and_trim_video(video_path, video_path[:-4] + "_cropped.mp4", DATA['crop_box'], 0, clip.duration)

        if self.pose_estimation:
            DATA['pose_estimation'] = ClipProcessor.process_pose(video_path = video_path, clip_start = 0, clip_end = clip.duration)

        return DATA

    def execute(self):
        
        clips_to_process = self.fetch_remaining_clips()
        print(clips_to_process)
        
        for name in clips_to_process:
            
            try:
                DATA = self.analyze_clip(self.clip_folder + name + ".mp4")

                crop_data = ""
                attribute_data = ""

                if self.crop: crop_data = DATA['crop_box']
                if self.attribute: attribute_data = DATA['text_attributes']

                columns_to_extract = ['CLIP ID', 'VIDEO ID', 'START TIME', 'END TIME']  # Modify as needed

                row = self.clip_csv_pd.loc[self.clip_csv_pd['CLIP ID'] == name, columns_to_extract]
                old_record = row.values.tolist()[0]

                language_value = self.video_csv_pd.loc[self.video_csv_pd['VIDEO ID'] == row['VIDEO ID'].values[0], 'Language'].iloc[0]
                

                record = pd.Series(old_record + [crop_data, attribute_data] + [language_value],
                    index = ['CLIP ID','VIDEO ID','START TIME','END TIME', 'Crop Box', 'Attributes', 'Language'])
                print("adding row: ")
                print(record)
                
                self.feature_csv_pd.loc[len(self.feature_csv_pd)] = record  # adding a row

                # self.video_csv is the PATH
                self.feature_csv_pd.to_csv(self.feature_csv, index=False)

                if self.pose_estimation:
                    DATA['pose_estimation'].to_csv(self.pose_folder + name + ".csv")


                # MARK CLIP PROCESSED
                self.clip_csv_pd.loc[self.clip_csv_pd['CLIP ID'] == name, 'PROCESSED?'] = 1

                # self.query_csv is the PATH
                self.clip_csv_pd.to_csv(self.clip_csv, index=False) 
            except Exception as e:
                print("SYSTEM: Ruh Roh. Problem analyzing this clip: ")
                print(e)
                continue;

    




# PORT LATER
# # this function will perform the cropping + saving
                # results = ClipProcessor.process(video_path = video_path, clip_start = start_time, clip_end = end_time)

                # crop_box = ClipProcessor.find_furthest_points(results['grouped_bbox'])

                # ClipProcessor.crop_and_trim_video(video_path = video_path,
                #                                 output_name = output_filename,
                #                                 bbox = crop_box,
                #                                 clip_start = start_time,
                #                                 clip_end = end_time                      
                #                                 )

