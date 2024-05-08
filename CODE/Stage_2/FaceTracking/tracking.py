from .video_processor import process_video
from .clustering_algos import *
from .grouping import group_bbox, group_probabilities, process_predictions
import cv2
from .load_model import Predictor
import numpy as np


class VideoProcessor:
       def __init__(self):
              self.face_attributes = ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
                     'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                     'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Gray_Hair',
                     'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                     'Mustache', 'Narrow_Eyes', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                     'Receding_Hairline', 'Rosy_Cheeks', 'Smiling', 'Straight_Hair',
                     'Wavy_Hair', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necktie',
                     'Young', 'Facial_Hair']


              self.attr_model = Predictor('CODE/Stage_2/FaceTracking/models/epoch_9_loss_14.826.pth', len(self.face_attributes), device='cuda')

              # self.attr_model = Predictor('FaceTracking/models/epoch_9_loss_14.826.pth', len(self.face_attributes), device='cuda')


       def proccess_video(self, video_path):
              frames, embeddings, predictions, face_counts = process_video(video_path, self.attr_model)
              min_num_faces = round(np.percentile(face_counts, 5))
              min_num_faces = max(min_num_faces, 2)

              print(frames)
              print(embeddings)
              print(predictions)
              print(min_num_faces)

              labels = cluster_hdbscan(embeddings, min_num_faces, max(len(frames)//5, 1), plot_data=False)

              print(labels)
              grouped_bbox = group_bbox(frames, labels)
              average_preds = group_probabilities(predictions, labels)


              text_attributes = process_predictions(average_preds, self.face_attributes)
              return grouped_bbox, text_attributes
       def visualize(self, video_path, bounding_boxes, output_video_path, predictions=None):
              # Load video
              cap = cv2.VideoCapture(video_path)
              if not cap.isOpened():
                     print("Error: Unable to open video.")
                     return

              # Get video properties
              fps = cap.get(cv2.CAP_PROP_FPS)
              width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
              height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

              # Define colors for different IDs
              colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

              # Define the codec and create VideoWriter object
              fourcc = cv2.VideoWriter_fourcc(*'mp4v')
              out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

              # Process each frame
              frame_index = 0
              while cap.isOpened():
                     ret, frame = cap.read()
                     if not ret:
                            break

                     # frame = imutils.resize(frame, width=1024)

                     # Draw bounding boxes and predictions for each ID
                     for id, boxes in bounding_boxes.items():
                            if frame_index in boxes:
                                   box = boxes[frame_index]
                                   color = colors[id % len(colors)]  # Choose color based on ID
                                   x, y, w, h = box
                                   cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                                   # Add predictions if available
                                   if predictions and id in predictions:
                                          pred_lines = predictions[id]
                                          y_offset = y - 10
                                          for line in pred_lines:
                                                 cv2.putText(frame, line, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                             color, 2)
                                                 y_offset -= 20  # Adjust the offset for the next line

                     frame = cv2.resize(frame, (width, height))
                     # Write the frame into the output video
                     out.write(frame)
                     cv2.imwrite('media/sample_img.png', frame)

                     frame_index += 1

              # Release resources
              cap.release()
              out.release()
              cv2.destroyAllWindows()
