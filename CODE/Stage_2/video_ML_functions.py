import cv2
import numpy as np
import cv2
import numpy as np
import os
from retinaface import RetinaFace
import matplotlib.pyplot as plt

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


def detect_faces_retinaface(retina_face_object, np_image: np.ndarray, plot: bool = False):
    """
    Detect faces in a NumPy image array using RetinaFace directly on numpy arrays,
    optionally plot the results, and return bounding boxes.

    :param np_image: Input image as a NumPy array.
    :param plot: Boolean flag to indicate whether to plot the detection results.
    :return: List of bounding boxes for each detected face, each box is a tuple (x, y, width, height).
    """
    # Initialize the RetinaFace detector
    app = retina_face_object; 
    # app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'recognition'])
    # app.prepare(ctx_id=0, det_size=(640, 640))


    # Detect faces
    faces = app.get(np_image)

    # Extract bounding boxes
    bboxes = [(int(face.bbox[0]), int(face.bbox[1]), int(face.bbox[2] - face.bbox[0]), int(face.bbox[3] - face.bbox[1])) for face in faces]

    # if plot:
    #     # If plotting is enabled, display the image with detected faces
    #     for bbox in bboxes:
    #         cv2.rectangle(np_image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        # cv2.imshow('Faces', np_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return bboxes



import numpy as np
import random

def single_face_and_size_checker(frame_range: tuple, custom_object, num_samples: int, min_size: int):
    """
    Check if randomly sampled frames from a video meet the criteria:
    - Exactly one face is detected
    - The detected face is larger than a specified minimum size in pixels (width/height)

    :param frame_range: A tuple containing the start and end frames (start_frame, end_frame).
    :param custom_object: An object with a .get_frame(num) function that returns a NumPy image of a frame.
    :param num_samples: Number of frames to randomly sample.
    :param min_size: Minimum size (either width or height) required for the detected face.
    :return: True if all criteria are met for all sampled frames, otherwise False.
    """
    start_frame, end_frame = frame_range
    # Ensure we don't sample more frames than available in the range
    max_samples = min(num_samples, end_frame - start_frame + 1)
    sample_frames = random.sample(range(start_frame, end_frame + 1), max_samples)

    # RETINAFACE ONLY
    app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    for frame_num in sample_frames:
        np_image = custom_object.get_frame(frame_num)

        

        bboxes = detect_faces_retinaface(app, np_image)

        # Check if exactly one face is detected and its size is larger than the specified min_size
        if len(bboxes) != 1 or (bboxes[0][2] < min_size and bboxes[0][3] < min_size):
            return False

    return True

def detect_faces_ssd(np_image: np.ndarray, confidence_threshold: float = 0.5):
    """
    Detect faces in a NumPy image array using SSD MobileNet, without plotting.

    :param np_image: Input image as a NumPy array.
    :param confidence_threshold: Confidence threshold for detected faces.
    :return: List of bounding boxes for each detected face, each box is a tuple (x, y, width, height).
    """
    # Load the pre-trained model and its weights (make sure to download these files and adjust paths as necessary)
    model_path = "deploy.prototxt.txt"  # Model configuration file
    weights_path = "res10_300x300_ssd_iter_140000.caffemodel"  # Model weights file
    net = cv2.dnn.readNetFromCaffe(model_path, weights_path)

    # Prepare the image for detection
    (h, w) = np_image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(np_image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the detections
    net.setInput(blob)
    detections = net.forward()

    bboxes = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the threshold
        if confidence > confidence_threshold:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Add the bounding box to the list
            bboxes.append((startX, startY, endX - startX, endY - startY))

    return bboxes
import cv2
import mediapipe as mp
import numpy as np

# Function to detect facial landmarks using MediaPipe FaceMesh
def detect_facial_landmarks(np_image):
    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0)

    # Convert the color space from BGR (OpenCV default) to RGB
    image_rgb = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    # Process the image to detect facial landmarks
    results = face_mesh.process(image_rgb)

    # Extract facial landmarks
    if results.multi_face_landmarks:
        # Assuming a single face detection, extract the first face's landmarks
        landmarks = results.multi_face_landmarks[0].landmark
        # Convert landmarks to a list of tuples (x, y, z)
        landmarks_list = [(landmark.x, landmark.y, landmark.z) for landmark in landmarks]
        return landmarks_list
    else:
        return None

from tqdm import tqdm
import gc  # Import the garbage collection module

def calculate_mouth_opening(landmarks, upper_lip_idx, lower_lip_idx):
    # Extract the landmarks for the specified upper and lower lip
    upper_lip = landmarks[upper_lip_idx]
    lower_lip = landmarks[lower_lip_idx]

    # Calculate the distance between the upper and lower lip landmarks
    # Since the landmarks are normalized, we're ignoring the z-coordinate
    mouth_opening = ((upper_lip[0] - lower_lip[0]) ** 2 + (upper_lip[1] - lower_lip[1]) ** 2) ** 0.5

    return mouth_opening
def detect_speaking(mouth_opening_values):
    """
    Determines if a person was speaking based on a series of mouth opening values.

    Args:
        mouth_opening_values (list): A list of mouth opening measurements.

    Returns:
        bool: True if the person was likely speaking, False otherwise.
    """
    if not mouth_opening_values:
        return False
    print(mouth_opening_values)
    # Calculate the average mouth opening
    average_opening = sum(mouth_opening_values) / len(mouth_opening_values)

    # Define a threshold for significant deviation (this may need tuning)
    threshold = average_opening * 1.05  # 20% above the average as an example

    # Count the number of significant movements
    significant_movements = sum(opening > threshold for opening in mouth_opening_values)

    # Define a criterion for speaking (this is arbitrary and may need adjustment)

    return significant_movements > 5
    # speaking_criteria = len(mouth_opening_values) * 0.  # Speaking 30% of the time

    # # Determine if speaking
    # is_speaking = significant_movements > speaking_criteria

    # return is_speaking


def analyze_speaking(test, start_frame, end_frame, upper_lip_idx, lower_lip_idx, frame_step=1, batch_size=30):


    total_frames = range(start_frame, end_frame + 1, frame_step)
    num_batches = len(total_frames) // batch_size + (1 if len(total_frames) % batch_size != 0 else 0)

    for batch in range(num_batches):
        try:
        
            if batch >= 10: break;

            mouth_openings = []
            start = batch * batch_size
            end = min((batch + 1) * batch_size, len(total_frames))
            batch_frames = total_frames[start:end]

            with tqdm(total=len(batch_frames), desc=f"Batch {batch+1}/{num_batches}") as pbar:
                for frame_number in batch_frames:
                    np_image = test.get_frame(frame_number)
                    if np_image is not None:
                        landmarks = detect_facial_landmarks(np_image)
                        if landmarks:
                            mouth_opening = calculate_mouth_opening(landmarks, upper_lip_idx, lower_lip_idx)
                            mouth_openings.append(mouth_opening)
                    # Update the progress bar by one step
                    pbar.update(1)

            # Explicitly delete variables that are no longer needed and clear the list to free up memory
            del np_image, landmarks

            # Call the garbage collector manually to force cleanup
            gc.collect()

            if detect_speaking(mouth_openings):
                return True;

        except Exception(e):
            print(e)
            continue;

    return False


#     # Detect speaking based on the calculated mouth openings
#     is_speaking = detect_speaking(mouth_openings)

#     # Further cleanup after processing is done
#     del mouth_openings
#     gc.collect()

#     return is_speaking
