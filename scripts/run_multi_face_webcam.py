"""
run_multi_face_webcam.py

Overview:
Run a live multi-face webcam pipeline that:
assigns persistent IDs for multiple faces and computs the mouth ratio for each, 
pluse stores mouth history of 15 frames to estimate a speaking score. 
    - these are set in the config file and are currently 4 faces and 15 frames
Ultimately highlights the active speaker.
"""

import cv2
import mediapipe as mp

from src.config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    MAX_NUM_FACES,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_MATCH_DISTANCE,
    HISTORY_SIZE,
)
from src.features.mouth_features import compute_mouth_open_ratio
from src.tracking.track_utils import get_face_bbox, get_bbox_center
from src.tracking.face_tracker import FaceTracker


def main():
    """
    run main live function 

    1. open webcam
    2. initialize face mesh 
    3. detect multiple faces in each frame 
    4. compute mouth ratio for each face detected 
    5. update persistent face tracks 
    6. estimate a speaking score for each tracked face
    7. highlight the speaking face as active
    """
    # ----------------------------------------
    # Load the face mesh  
    # ----------------------------------------
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    # ----------------------------------------
    # set frame size and configs 
    # ----------------------------------------
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    tracker = FaceTracker(
        max_match_distance=MAX_MATCH_DISTANCE,
        history_size=HISTORY_SIZE,
    )

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=MAX_NUM_FACES,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as face_mesh:

        while True:
            success, frame = cap.read() # read one frame from webcam
            if not success:
                print("Error: Failed to read frame.") # stop if cannot read
                break

    # ----------------------------------------
    # preprocess the frame 
    # ----------------------------------------
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            detections = []
            # if face detected, process each one 
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    bbox = get_face_bbox(face_landmarks, frame_width, frame_height)
                    center = get_bbox_center(bbox)
                    mouth_ratio = compute_mouth_open_ratio(face_landmarks)

                    detections.append(
                        {
                            "bbox": bbox,
                            "center": center,
                            "mouth_ratio": mouth_ratio,
                        }
                    )

    # ----------------------------------------
    # update the persistent IDs 
    # ----------------------------------------
            tracks = tracker.update(detections)

            active_speaker_id = None
            best_score = -1.0

            for track_id, track in tracks.items():
                if track["speaking_score"] > best_score:
                    best_score = track["speaking_score"]
                    active_speaker_id = track_id

            for track_id, track in tracks.items():
                x_min, y_min, x_max, y_max = track["bbox"]

                label = f"ID {track_id} | Score: {track['speaking_score']:.4f}"

                if track_id == active_speaker_id:
                    label += " | ACTIVE"
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x_min, max(y_min - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            cv2.imshow("Multi-Face Speaker Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()