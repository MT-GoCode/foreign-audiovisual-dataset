from tracking import VideoProcessor

video_path = 'media/0o91OXXAthXXoGK.mp4'
output_video_path = 'media/out_test.mp4'


processor = VideoProcessor()

grouped_bbox, text_attributes = processor.proccess_video(video_path)

print(grouped_bbox, text_attributes)

processor.visualize(video_path, grouped_bbox, output_video_path, text_attributes)

