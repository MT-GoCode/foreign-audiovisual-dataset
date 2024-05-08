from mtcnn import MTCNN
from deepface import DeepFace
import cv2
import numpy as np
import imutils
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity



detector = MTCNN()

embeddings = []
colors = {}
rectangles = {}

cap = cv2.VideoCapture('media/diversity_stock.mp4')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create VideoWriter object to save the processed video
out = cv2.VideoWriter('media/output_tracker.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

i = 0
face_idx = 0
all_embeds = np.zeros((10, 512))

# Get total frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

all_embeddings = []

# Use tqdm to create a progress bar
with tqdm(total=total_frames) as pbar:
    while cap.isOpened():
        i += 1
        ret, frame = cap.read()

        if not ret or i > 100:
            break

        frame = imutils.resize(frame, width=1024)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if i % 1 == 0:
            detections = detector.detect_faces(frame)
            print(len(detections))

            for detection in detections:
                confidence = detection["confidence"]
                if confidence > 0.8:
                    x, y, w, h = detection["box"]
                    # Draw a rectangle around the detected face
                    detected_face = frame[int(y):int(y+h), int(x):int(x+w)]

                    embedding = np.array(DeepFace.represent(detected_face, model_name='Facenet', enforce_detection=False)[0]['embedding'])
                    similarities = cosine_similarity(all_embeds, embedding.reshape(1, -1))
                    print(np.max(similarities))
                    if np.max(similarities) > 0.80:
                        closest_index = np.argmax(similarities)
                        # all_embeds[np.argmax(similarities)] = embedding
                        color = colors[closest_index]
                        # Update rectangle coordinates
                        rectangles[closest_index] = (x, y, w, h)

                    else:
                        all_embeds[face_idx] = embedding
                        colors[face_idx] = tuple(np.random.choice(range(256), size=3).tolist())
                        color = colors[face_idx]
                        rectangles[face_idx] = (x, y, w, h)
                        face_idx = (face_idx + 1)%all_embeds.shape[0]

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)  # You can adjust the color and thickness of the rectangle

        else:
            # Draw previously detected rectangles
            for idx, rect in rectangles.items():
                x, y, w, h = rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), colors[idx], 2)

        out.write(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (frame_width, frame_height)))

        # Update progress bar
        pbar.update(1)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
