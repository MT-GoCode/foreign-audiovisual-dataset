import argparse
import yaml

from CODE.Stage_1.stage_1_download import *
from CODE.Stage_2.stage_2_video_filtering import *
from CODE.Stage_3.stage_3_analysis import *
from CODE.Post.post_processing import *


def load_configuration(file_path):
    """ Load YAML configuration from the file specified. """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_path):
    # Load the configuration from the YAML file
    config = load_configuration(config_path)
    pipeline = config['Collection_Pipeline']

    if config['infrastructure'] == 'local':
        
        if pipeline['download']['enabled']:
            Stage_1 = LocalDownloader(query_csv = pipeline['download']['query_csv'],
                                    raw_folder = pipeline['download']['raw_folder'],
                                    video_csv = pipeline['download']['video_csv'],
                                    results_per_query = pipeline['download']['results_per_query'],
                                    limit_query = pipeline['download']['limit_query'],
                                    limit_videos = pipeline['download']['limit_videos'])
            
            Stage_1.execute()

        if pipeline['filter']['enabled']:

            
            Stage_2 = LocalFilter(
                            # a couple parameters overlap from download stage
                            raw_folder = pipeline['download']['raw_folder'],
                            video_csv = pipeline['download']['video_csv'],

                            clip_folder = pipeline['filter']['clip_folder'],
                            clip_csv = pipeline['filter']['clip_csv'],
                            min_face_size = int(pipeline['filter']['min_face_size']),
                            skip_crop = bool(pipeline['filter']["skip_crop"]),
                            min_length = float(pipeline['filter']["min_length"]),
                            limit_videos = int(pipeline['filter']['limit_videos']),
                            voice_detection_smoothing = float(pipeline['filter']['limit_videos']))
            print ("obj creation")
            Stage_2.execute()

        if pipeline['analysis']['enabled']:
            
            if bool(pipeline['analysis']['attribute']):

                Stage_3 = LocalAnalyzer(
                    clip_folder = pipeline['filter']['clip_folder'],
                    clip_csv = pipeline['filter']['clip_csv'],

                    video_csv = pipeline['download']['video_csv'], # just to get language


                    limit_videos = pipeline['analysis']['limit_videos'],
                    feature_csv = pipeline['analysis']['feature_csv'],

                    crop = pipeline['analysis']['crop'],
                    download_crops = pipeline['analysis']['download_crops'],
                    
                    attribute = pipeline['analysis']['attribute'],

                    pose_estimation = pipeline['analysis']['pose_estimation'],
                    
                    pose_folder = pipeline['analysis']['pose_folder'],
                )

                Stage_3.execute()

        if pipeline['post_processing']['enabled']:

            if pipeline['post_processing']['plot']:
                P = Plotter(feature_csv = pipeline['analysis']['feature_csv'],
                            plot_folder = pipeline['post_processing']['plot_folder'])
                
                P.execute()

            # Stage_3 = LocalCropper(
            #     crop = 
            #     download_crops = bool(pipeline['analysis']['crop'])
                
            #     limit_videos = int(pipeline['analysis']['limit_videos'])

            # )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    
    args = parser.parse_args()

    # Call the main function with the path to the YAML configuration file
    main(args.config)