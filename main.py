import argparse
import yaml

from CODE.Stage_1.stage_1_download import *
from CODE.Stage_2.stage_2_video_filtering import *


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

        if pipeline['filtering']['enabled']:

            
            Stage_2 = LocalFilter(
                            # a couple parameters overlap from download stage
                            raw_folder = pipeline['download']['raw_folder'],
                            video_csv = pipeline['download']['video_csv'],

                            clip_folder = pipeline['filtering']['clip_folder'],
                            clip_csv = pipeline['filtering']['clip_csv'],
                            min_face_size = int(pipeline['filtering']['min_face_size']),
                            skip_crop = bool(pipeline['filtering']["skip_crop"]),
                            min_length = float(pipeline['filtering']["min_length"]),
                            limit_videos = int(pipeline['filtering']['limit_videos']),
                            voice_detection_smoothing = float(pipeline['filtering']['limit_videos']))
            print ("obj creation")
            Stage_2.execute()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    
    args = parser.parse_args()

    # Call the main function with the path to the YAML configuration file
    main(args.config)