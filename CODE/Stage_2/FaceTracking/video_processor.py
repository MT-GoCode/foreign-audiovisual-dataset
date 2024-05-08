from mtcnn import MTCNN
from deepface import DeepFace
import cv2
from tqdm import tqdm

import numpy as np



detector = MTCNN()

def process_video(path, prediction_model, step=24, end=None, pred_freq = .2):


    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
      raise Exception("Invalid Video: ", path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = min(total_frames//4, step)
    print(step)

    frame_count = 0
    pred_idx = 0

    all_embeddings = []
    predictions = {}
    history = {}
    face_counts = []

    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            # if i<cap.get(cv2.CAP_PROP_FPS)*32:
            #     continue
            if not ret or (end and frame_count>cap.get(cv2.CAP_PROP_FPS)*end):
                break

            # frame = imutils.resize(frame, width=1024)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(frame)
            face_counts.append(0)
            history[frame_count] = []
            for detection in detections:
                confidence = detection["confidence"]
                if confidence > 0.9:
                    x, y, w, h = detection["box"]
                    if (h*w)<1200:
                        continue


                    detected_face = frame[int(y):int(y+h), int(x):int(x+w)]



                    embedding = np.array(DeepFace.represent(detected_face, model_name='Facenet512', enforce_detection=False)[0]['embedding'])
                    all_embeddings.append(embedding)
                    if np.random.random() < pred_freq or pred_idx == 0:
                        predictions[pred_idx] = prediction_model.predict(detected_face)
                    face_counts[-1] += 1
                    history[frame_count].append((pred_idx, (x, y, w, h)))
                    pred_idx += 1

            pbar.update(step)
            frame_count += step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    all_embeddings = np.array(all_embeddings)

    return history, all_embeddings, predictions, face_counts




