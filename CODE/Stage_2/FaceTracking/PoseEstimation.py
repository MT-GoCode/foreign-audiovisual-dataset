"""Estimate head pose according to the facial landmarks"""
import cv2
import numpy as np

pose_asset_path = "pose-assets/assets/"
class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, image_width, image_height):
        """Init a pose estimator.

        Args:
            image_width (int): input image width
            image_height (int): input image height
        """
        self.size = (image_height, image_width)
        self.model_points_68 = self._get_full_model_points()

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])

    def _get_full_model_points(self, filename=pose_asset_path + 'model.txt'):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # Transform the model into a front view.
        model_points[:, 2] *= -1

        return model_points

    def solve(self, points):
        """Solve pose with all the 68 image points
        Args:
            points (np.ndarray): points on image.

        Returns:
            Tuple: (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68, points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68,
            points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def visualize(self, image, pose, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        rotation_vector, translation_vector = pose
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

    def draw_axes(self, img, pose):
        R, t = pose
        img = cv2.drawFrameAxes(img, self.camera_matrix,
                                self.dist_coeefs, R, t, 30)

    def show_3d_model(self):
        from matplotlib import pyplot
        from mpl_toolkits.mplot3d import Axes3D
        fig = pyplot.figure()
        ax = Axes3D(fig)

        x = self.model_points_68[:, 0]
        y = self.model_points_68[:, 1]
        z = self.model_points_68[:, 2]

        ax.scatter(x, y, z)
        ax.axis('square')
        pyplot.xlabel('x')
        pyplot.ylabel('y')
        pyplot.show()


"""This module provides a face detection implementation backed by SCRFD.
https://github.com/deepinsight/insightface/tree/master/detection/scrfd
"""
import os

import cv2
import numpy as np
import onnxruntime


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class FaceDetector:

    def __init__(self, model_file):
        """Initialize a face detector.

        Args:
            model_file (str): ONNX model file path.
        """
        assert os.path.exists(model_file), f"File not found: {model_file}"

        self.center_cache = {}
        self.nms_threshold = 0.4
        self.session = onnxruntime.InferenceSession(
            model_file, providers=['CPUExecutionProvider'])

        # Get model configurations from the model file.
        # What is the input like?
        input_cfg = self.session.get_inputs()[0]
        input_name = input_cfg.name
        input_shape = input_cfg.shape
        self.input_size = tuple(input_shape[2:4][::-1])

        # How about the outputs?
        outputs = self.session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names

        # And any key points?
        self._with_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1

        if len(outputs) == 6:
            self._offset = 3
            self._strides = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self._offset = 3
            self._strides = [8, 16, 32]
            self._num_anchors = 2
            self._with_kps = True
        elif len(outputs) == 10:
            self._offset = 5
            self._strides = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self._offset = 5
            self._strides = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self._with_kps = True

    def _preprocess(self, image):
        inputs = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = inputs - np.array([127.5, 127.5, 127.5])
        inputs = inputs / 128
        inputs = np.expand_dims(inputs, axis=0)
        inputs = np.transpose(inputs, [0, 3, 1, 2])

        return inputs.astype(np.float32)

    def forward(self, img, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []

        inputs = self._preprocess(img)
        predictions = self.session.run(
            self.output_names, {self.input_name: inputs})

        input_height = inputs.shape[2]
        input_width = inputs.shape[3]
        offset = self._offset

        for idx, stride in enumerate(self._strides):
            scores_pred = predictions[idx]
            bbox_preds = predictions[idx + offset] * stride
            if self._with_kps:
                kps_preds = predictions[idx + offset * 2] * stride

            # Generate the anchors.
            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)

            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # solution-3:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))

                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))

                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

                # solution-1, c style:
                # anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                # for i in range(height):
                #    anchor_centers[i, :, 1] = i
                # for i in range(width):
                #    anchor_centers[:, i, 0] = i

                # solution-2:
                # ax = np.arange(width, dtype=np.float32)
                # ay = np.arange(height, dtype=np.float32)
                # xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                # anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

            # Filter the results by scores and threshold.
            pos_inds = np.where(scores_pred >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores_pred[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            if self._with_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list

    def _nms(self, detections):
        """None max suppression."""
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        scores = detections[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            _x1 = np.maximum(x1[i], x1[order[1:]])
            _y1 = np.maximum(y1[i], y1[order[1:]])
            _x2 = np.minimum(x2[i], x2[order[1:]])
            _y2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, _x2 - _x1 + 1)
            h = np.maximum(0.0, _y2 - _y1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= self.nms_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, img, threshold=0.5, input_size=None, max_num=0, metric='default'):
        input_size = self.input_size if input_size is None else input_size

        # Rescale the image?
        img_height, img_width, _ = img.shape
        ratio_img = float(img_height) / img_width

        input_width, input_height = input_size
        ratio_model = float(input_height) / input_width

        if ratio_img > ratio_model:
            new_height = input_height
            new_width = int(new_height / ratio_img)
        else:
            new_width = input_width
            new_height = int(new_width * ratio_img)

        det_scale = float(new_height) / img_height
        resized_img = cv2.resize(img, (new_width, new_height))

        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, threshold)
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        bboxes = np.vstack(bboxes_list) / det_scale

        if self._with_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        keep = self._nms(pre_det)

        det = pre_det[keep, :]

        if self._with_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]])

            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            if metric == 'max':
                values = area
            else:
                # some extra weight on the centering
                values = area - offset_dist_squared * 2.0

            # some extra weight on the centering
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]

            if kpss is not None:
                kpss = kpss[bindex, :]

        return det, kpss

    def visualize(self, image, results, box_color=(0, 255, 0), text_color=(0, 0, 0)):
        """Visualize the detection results.

        Args:
            image (np.ndarray): image to draw marks on.
            results (np.ndarray): face detection results.
            box_color (tuple, optional): color of the face box. Defaults to (0, 255, 0).
            text_color (tuple, optional): color of the face marks (5 points). Defaults to (0, 0, 255).
        """
        for det in results:
            bbox = det[0:4].astype(np.int32)
            conf = det[-1]
            cv2.rectangle(image, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), box_color)
            label = f"face: {conf:.2f}"
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (bbox[0], bbox[1] - label_size[1]),
                          (bbox[2], bbox[1] + base_line), box_color, cv2.FILLED)
            cv2.putText(image, label, (bbox[0], bbox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

def refine(boxes, max_width, max_height, shift=0.1):
    """Refine the face boxes to suit the face landmark detection's needs.

    Args:
        boxes: [[x1, y1, x2, y2], ...]
        max_width: Value larger than this will be clipped.
        max_height: Value larger than this will be clipped.
        shift (float, optional): How much to shift the face box down. Defaults to 0.1.

    Returns:
       Refined results.
    """
    refined = boxes.copy()
    width = refined[:, 2] - refined[:, 0]
    height = refined[:, 3] - refined[:, 1]

    # Move the boxes in Y direction
    shift = height * shift
    refined[:, 1] += shift
    refined[:, 3] += shift
    center_x = (refined[:, 0] + refined[:, 2]) / 2
    center_y = (refined[:, 1] + refined[:, 3]) / 2

    # Make the boxes squares
    square_sizes = np.maximum(width, height)
    refined[:, 0] = center_x - square_sizes / 2
    refined[:, 1] = center_y - square_sizes / 2
    refined[:, 2] = center_x + square_sizes / 2
    refined[:, 3] = center_y + square_sizes / 2

    # Clip the boxes for safety
    refined[:, 0] = np.clip(refined[:, 0], 0, max_width)
    refined[:, 1] = np.clip(refined[:, 1], 0, max_height)
    refined[:, 2] = np.clip(refined[:, 2], 0, max_width)
    refined[:, 3] = np.clip(refined[:, 3], 0, max_height)

    return refined
"""Human facial landmark detector based on Convolutional Neural Network."""
import os

import cv2
import numpy as np
import onnxruntime as ort


class MarkDetector:
    """Facial landmark detector by Convolutional Neural Network"""

    def __init__(self, model_file):
        """Initialize a mark detector.

        Args:
            model_file (str): ONNX model path.
        """
        assert os.path.exists(model_file), f"File not found: {model_file}"
        self._input_size = 128
        self.model = ort.InferenceSession(
            model_file, providers=["CPUExecutionProvider"])

    def _preprocess(self, bgrs):
        """Preprocess the inputs to meet the model's needs.

        Args:
            bgrs (np.ndarray): a list of input images in BGR format.

        Returns:
            tf.Tensor: a tensor
        """
        rgbs = []
        for img in bgrs:
            img = cv2.resize(img, (self._input_size, self._input_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgbs.append(img)

        return rgbs

    def detect(self, images):
        """Detect facial marks from an face image.

        Args:
            images: a list of face images.

        Returns:
            marks: the facial marks as a numpy array of shape [Batch, 68*2].
        """
        inputs = self._preprocess(images)
        marks = self.model.run(["dense_1"], {"image_input": inputs})
        return np.array(marks)

    def visualize(self, image, marks, color=(255, 255, 255)):
        """Draw mark points on image"""
        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(
                mark[1])), 1, color, -1, cv2.LINE_AA)


import cv2
import csv
import pandas as pd

def run(video_src, start_frame = None, end_frame = None):
    if start_frame ==None: start_frame = 0
    if end_frame == None: end_frame = 1E9

    cap = cv2.VideoCapture(video_src)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_detector = FaceDetector(pose_asset_path + "face_detector.onnx")
    mark_detector = MarkDetector(pose_asset_path + "face_landmarks.onnx")
    pose_estimator = PoseEstimator(frame_width, frame_height)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Initialize an empty list to store the results
    results = []

    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        
        if not ret:
            break  # Break the loop if there are no frames to read (end of video)

        faces, _ = face_detector.detect(frame, 0.7)

        if len(faces) > 0:
            face = refine(faces, frame_width, frame_height, 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]

            marks = mark_detector.detect([patch])[0].reshape([68, 2])
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            pose = pose_estimator.solve(marks)
            # Add the results to the list
            results.append([current_frame, pose[0].ravel().tolist()])

        current_frame += 1

        if cv2.waitKey(1) == 27:  # Allow early exit using ESC key
            break

    cap.release()
    
    cv2.destroyAllWindows()

    # Convert the results list to a DataFrame
    df_results = pd.DataFrame(results, columns=["Frame Number", "Rotation Vector"])
    return df_results

# # Example usage
# video_file = r"C:\Users\tminh\Downloads\DatasetGenerator\DATA\video_clips\test_cropped.mp4"
#  # Replace with your video file path
# start_frame = 0  # Replace with your desired start frame
# end_frame = 400  # Replace with your desired end frame
# print(run(video_file, start_frame, end_frame))