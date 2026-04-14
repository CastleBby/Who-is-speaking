"""
run_multi_face_video.py

essentially the same as run_multi_face_webcame, but reads a file path instead of 
VideoCapture(0)

OVERLAP DETECTION
compute the highest speaking score, 2nd, etc. if 1 or more over threshold flag overlap 

USAGE:

Example:
    PYTHONPATH=. python scripts/run_multi_face_video.py

Output:
- A video window showing:
  - bounding boxes for tracked faces
  - persistent IDs
  - speaking scores
  - ACTIVE label for the face with the highest current speaking score
"""

import os

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
    ACTIVE_SPEAKER_THRESHOLD,
    SPEAKER_MARGIN,
    WANTS_TO_SPEAK_RATIO, 
    WANTS_TO_SPEAK_THRESHOLD,
)
from src.features.mouth_features import compute_mouth_open_ratio
from src.tracking.track_utils import get_face_bbox, get_bbox_center
from src.tracking.face_tracker import FaceTracker


# --------------------------------------------------
# Set the input video path here
# --------------------------------------------------
# Change this path to match the location of your test video.
VIDEO_PATH = "data/raw/sample_videos/test_multi_face2.mp4"


def main():
    """
    Run the multi-face speaker-tracking pipeline on a prerecorded video.

    This function:
    1. opens the input video file,
    2. initializes MediaPipe Face Mesh,
    3. detects multiple faces in each frame,
    4. computes a mouth-open ratio for each detected face,
    5. updates persistent face tracks,
    6. estimates a speaking score for each tracked face,
    7. highlights the face with the highest score as ACTIVE,
    8. exits when the video ends or the user presses 'q'.

    Returns
    -------
    None
    """

    # ----------------------------------------
    # Check that the video file exists
    # ----------------------------------------
    # Stop early if the input path is wrong.
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found: {VIDEO_PATH}")
        return

    # ----------------------------------------
    # Load the MediaPipe Face Mesh solution
    # ----------------------------------------
    # This gives access to the face landmark detector.
    mp_face_mesh = mp.solutions.face_mesh

    # ----------------------------------------
    # Open the input video file
    # ----------------------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Stop immediately if the video cannot be opened.
    if not cap.isOpened():
        print(f"Error: Could not open video: {VIDEO_PATH}")
        return

    # ----------------------------------------
    # Initialize the custom face tracker
    # ----------------------------------------
    # The tracker manages persistent IDs and stores recent mouth-history
    # for speaking-score estimation.
    tracker = FaceTracker(
        max_match_distance=MAX_MATCH_DISTANCE,
        history_size=HISTORY_SIZE,
    )

    # ----------------------------------------
    # Initialize MediaPipe Face Mesh
    # ----------------------------------------
    # static_image_mode=False treats the input as a video stream.
    # max_num_faces limits how many faces can be detected per frame.
    # refine_landmarks=True improves detail around the mouth and eyes.
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=MAX_NUM_FACES,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as face_mesh:

        # ----------------------------------------
        # Main video processing loop
        # ----------------------------------------
        while True:
            # Read one frame from the video.
            success, frame = cap.read()

            # Stop when the video ends or a frame cannot be read.
            if not success:
                print("Finished reading video or failed to read frame.")
                break

            # ----------------------------------------
            # Resize the frame for consistency
            # ----------------------------------------
            # This keeps display size and processing scale more consistent
            # across different input videos.
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Extract frame dimensions after resizing.
            frame_height, frame_width = frame.shape[:2]

            # ----------------------------------------
            # Convert frame from BGR to RGB
            # ----------------------------------------
            # OpenCV uses BGR, while MediaPipe expects RGB.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run MediaPipe face landmark detection on the current frame.
            results = face_mesh.process(rgb_frame)
            num_faces = 0 if results.multi_face_landmarks is None else len(results.multi_face_landmarks)
            print(f"Detected faces: {num_faces}")

            # ----------------------------------------
            # Build current-frame detections
            # ----------------------------------------
            # Each detection stores:
            # - bounding box
            # - center point
            # - mouth-open ratio
            detections = []

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Estimate a face bounding box from the landmarks.
                    bbox = get_face_bbox(face_landmarks, frame_width, frame_height)

                    # Compute the center of the bounding box.
                    center = get_bbox_center(bbox)

                    # Compute the mouth-open ratio for this face.
                    mouth_ratio = compute_mouth_open_ratio(face_landmarks)

                    # Store this detection for tracker matching.
                    detections.append(
                        {
                            "bbox": bbox,
                            "center": center,
                            "mouth_ratio": mouth_ratio,
                        }
                    )

            # ----------------------------------------
            # Update persistent tracks
            # ----------------------------------------
            # The tracker matches current detections to prior tracks and:
            # - reuses IDs when matches are found
            # - creates new IDs when needed
            # - updates recent mouth-history
            # - computes speaking scores
            tracks = tracker.update(detections)

            # ----------------------------------------
            # Determine the current active speaker
            # ----------------------------------------
            # The active speaker is defined as the track with the highest
            # current speaking score.

            # This reduces false positives caused by tiny jitter or weak facial motion.
            active_speaker_id = None
            best_score = 0.0
            second_best_score = 0.0

            # Sort tracks by speaking score from highest to lowest.
            sorted_tracks = sorted(
                tracks.items(),
                key=lambda item: item[1]["speaking_score"],
                reverse=True,
            )

            if len(sorted_tracks) > 0:
                best_track_id, best_track = sorted_tracks[0]
                best_score = best_track["speaking_score"]

                second_best_score = (
                    sorted_tracks[1][1]["speaking_score"]
                    if len(sorted_tracks) > 1
                    else 0.0
                )

                # assign one clearly stronger than 2nd highest with margin 
                if (
                    best_score >= ACTIVE_SPEAKER_THRESHOLD
                    and (best_score -second_best_score) >= SPEAKER_MARGIN
                ):
                    active_speaker_id = best_track_id
                else:
                    active_speaker_id = None

            # ----------------------------------------
            # Determine whether any other tracked face appears to want to speak
            # ----------------------------------------
            # This is a softer rule than the ACTIVE speaker rule.
            # A face can be flagged as wanting to speak if:
            # 1. it is not the active speaker,
            # 2. its speaking score exceeds a smaller threshold,
            # 3. its score is reasonably close to the active speaker's score.
            wants_to_speak_ids = []

            if active_speaker_id is not None and best_score > 0:
                for track_id, track in tracks.items():
                    if track_id == active_speaker_id:
                        continue

                    contender_score = track["speaking_score"]

                    if (
                        contender_score >= WANTS_TO_SPEAK_THRESHOLD
                        and contender_score >= WANTS_TO_SPEAK_RATIO * best_score
                    ):
                        wants_to_speak_ids.append(track_id)


            # ----------------------------------------
            # Draw tracked faces and labels
            # ----------------------------------------
            for track_id, track in tracks.items():
                # Unpack the bounding box coordinates.
                x_min, y_min, x_max, y_max = track["bbox"]

                # Build the display label with ID and speaking score.
                label = f"ID {track_id} | Score: {track['speaking_score']:.4f}"

                # Highlight the highest-scoring face as ACTIVE.
                if track_id == active_speaker_id:
                    label += " | ACTIVE"
                    color = (0, 255, 0)

                elif track_id in wants_to_speak_ids:
                    label += "WANTS TO TALK"
                    color = (0, 255, 255) 
                
                else:
                    color = (255, 0, 0)
                

                # Draw the bounding box.
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                # Draw the text label above the box.
                cv2.putText(
                    frame,
                    label,
                    (x_min, max(y_min - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            # ----------------------------------------
            # Display the processed video frame
            # ----------------------------------------
            cv2.imshow("Multi-Face Speaker Tracking - Video", frame)

            # ----------------------------------------
            # Exit when the user presses 'q'
            # ----------------------------------------
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

    # ----------------------------------------
    # Clean up resources
    # ----------------------------------------
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Run the main pipeline only when this file is executed directly.
    main()