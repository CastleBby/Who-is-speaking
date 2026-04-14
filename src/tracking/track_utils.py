"""
track_utils.py

Overview:
Utility functions for estimating face position from MediaPipe landmarks
and comparing detected faces across frames
"""

import math


def get_face_bbox(face_landmarks, frame_width, frame_height):
    """
    Compute an approximate face bounding box from MediaPipe landmarks

    Returns
    -------
    tuple
        (x_min, y_min, x_max, y_max) bounding box in pixel coordinates
    """

    xs = [landmark.x * frame_width for landmark in face_landmarks.landmark]
    ys = [landmark.y * frame_height for landmark in face_landmarks.landmark]

    x_min = int(min(xs))
    y_min = int(min(ys))
    x_max = int(max(xs))
    y_max = int(max(ys))

    return x_min, y_min, x_max, y_max


def get_bbox_center(bbox):
    """
    Compute the center point of a bounding box

    Returns
    -------
    tuple
        (center_x, center_y)
    """

    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return center_x, center_y


def euclidean_distance(point1, point2):
    """
    Compute Euclidean distance between two 2D points
    """

    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)