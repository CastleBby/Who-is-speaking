"""
mouth_features.py

Overview:
Helper functions for measuring mouth movement from
MediaPipe Face Mesh landmarks. The main feature computed here is a
mouth-open ratio, aka a visual signal for how open
a person's mouth is in a given frame.

Usage:
Pass a MediaPipe face_landmarks object into compute_mouth_open_ratio().

Example:
    mouth_ratio = compute_mouth_open_ratio(face_landmarks)

Inputs:
- face_landmarks:
    A MediaPipe Face Mesh landmark object for one detected face.

Output:
- float:
    A normalized mouth-open ratio. Larger values generally indicate a
    more open mouth.

"""
import math


def euclidean_distance(p1, p2):
    # Compute the straight-line Euclidean distance between two landmark points
    # Return distance between p1 and p2 in normalized image space

    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def compute_mouth_open_ratio(face_landmarks):
    # Compute a normalized mouth opening ratio from one face landmarks
    # Ratio = vertical mouth opening / horizontal mouth width 
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    left_mouth = face_landmarks.landmark[61]
    right_mouth = face_landmarks.landmark[291]

    vertical = euclidean_distance(upper_lip, lower_lip)
    horizontal = euclidean_distance(left_mouth, right_mouth)

    if horizontal == 0:
        return 0.0

    return vertical / horizontal