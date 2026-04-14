"""
face_tracker.py

Overview:
A simple tracking by detection face tracker that assigns persistent IDs
to faces across frames using bounding-box center matching
"""

from collections import deque

from src.features.temporal_features import compute_motion_energy
from src.tracking.track_utils import euclidean_distance


class FaceTracker:
    """
    Simple multi-face tracker with persistent IDs and mouth history
    """

    def __init__(self, max_match_distance=80, history_size=15):
        self.max_match_distance = max_match_distance
        self.history_size = history_size
        self.next_id = 0
        self.tracks = {}

    def update(self, detections):
        """
        Update tracks using current-frame detections

        Returns
        -------
        dict
            Updated tracks dictionary.
        """

        updated_tracks = {}
        used_track_ids = set()

        for detection in detections:
            detection_center = detection["center"]

            best_track_id = None
            best_distance = float("inf")

            for track_id, track in self.tracks.items():
                if track_id in used_track_ids:
                    continue

                dist = euclidean_distance(detection_center, track["center"])

                if dist < best_distance and dist <= self.max_match_distance:
                    best_distance = dist
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = self.next_id
                self.next_id += 1

                updated_tracks[best_track_id] = {
                    "id": best_track_id,
                    "center": detection["center"],
                    "bbox": detection["bbox"],
                    "mouth_history": deque(maxlen=self.history_size),
                    "speaking_score": 0.0,
                }
            else:
                updated_tracks[best_track_id] = self.tracks[best_track_id]
                updated_tracks[best_track_id]["center"] = detection["center"]
                updated_tracks[best_track_id]["bbox"] = detection["bbox"]

            updated_tracks[best_track_id]["mouth_history"].append(detection["mouth_ratio"])
            used_track_ids.add(best_track_id)

        for track_id, track in updated_tracks.items():
            history = list(track["mouth_history"])

            if len(history) >= 2:
                import pandas as pd
                history_series = pd.Series(history)
                score_series = compute_motion_energy(history_series, window=min(5, len(history)))
                track["speaking_score"] = float(score_series.iloc[-1])
            else:
                track["speaking_score"] = 0.0

        self.tracks = updated_tracks
        return self.tracks