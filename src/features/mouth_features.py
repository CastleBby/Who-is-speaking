import math


def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def compute_mouth_open_ratio(face_landmarks):
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    left_mouth = face_landmarks.landmark[61]
    right_mouth = face_landmarks.landmark[291]

    vertical = euclidean_distance(upper_lip, lower_lip)
    horizontal = euclidean_distance(left_mouth, right_mouth)

    if horizontal == 0:
        return 0.0

    return vertical / horizontal