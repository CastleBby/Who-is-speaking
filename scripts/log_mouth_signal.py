"""
log_mouth_signal.py

Overview:
Script extends opens the webcam, detects a face with MediaPipe Face Mesh,
computes the mouth-open ratio for each frame, displays the value live, and
saves the results to a CSV file when the user exits.

Answers the following questions: 
- Is the signal stable during a neutral face?
- Does the signal rise and fall during mouth movement?
- Does normal speaking create a distinctive pattern over time?
- Is the feature too noisy to use directly?

Test ran: 25 seconds total test (approximately)
5 sec. neutral face
5 seconds exaggerated open/close mouth 
10 seconds of normal speech
5 seconds neutral face 

Usage:
    PYTHONPATH=. python scripts/log_mouth_signal.py

Output:
- A webcam window showing the face mesh and current mouth-open ratio
- A CSV file saved to outputs/logs/mouth_signal_log.csv

CSV Columns:
- frame_index: frame number in the session
- timestamp_sec: elapsed time in seconds since the session began
- mouth_ratio: computed mouth-open ratio for that frame
- face_detected: 1 if a face was detected, 0 otherwise
"""

import os
import time

import cv2
import mediapipe as mp
import pandas as pd

from src.config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    MAX_NUM_FACES,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)
from src.features.mouth_features import compute_mouth_open_ratio


def main():
    """
    Run the webcam mouth-signal logging pipeline

    This function:
    1. opens the webcam
    2. initializes MediaPipe Face Mesh
    3. reads frames continuously
    4. computes a mouth-open ratio for the first detected face
    5. logs frame-level results into a list
    6. displays the value live
    7. saves the log to CSV when the user presses 'q'

    Returns
    -------
    None
    """

    # ----------------------------------------
    # Create output directory if it does not exist
    # ----------------------------------------
    # This ensures the CSV save step will not fail due to a missing folder.
    output_dir = "outputs/logs"
    os.makedirs(output_dir, exist_ok=True)

    # Define the CSV output path.
    output_csv = os.path.join(output_dir, "mouth_signal_log.csv")

    # ----------------------------------------
    # Load MediaPipe helper modules
    # ----------------------------------------
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # ----------------------------------------
    # Open the default webcam
    # ----------------------------------------
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set the requested frame size from config.py.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # ----------------------------------------
    # Prepare data logging containers
    # ----------------------------------------
    # Each frame will produce one record stored as a dictionary.
    log_data = []

    # Record the start time so elapsed seconds can be computed per frame.
    start_time = time.time()

    # Track frame number manually.
    frame_index = 0

    # ----------------------------------------
    # Initialize MediaPipe Face Mesh
    # ----------------------------------------
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=MAX_NUM_FACES,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as face_mesh:

        # ----------------------------------------
        # Main webcam processing loop
        # ----------------------------------------
        while True:
            success, frame = cap.read()

            if not success:
                print("Error: Failed to read frame.")
                break

            # Increment frame counter for each successfully read frame.
            frame_index += 1

            # Compute elapsed time since the script started.
            timestamp_sec = time.time() - start_time

            # Flip horizontally for mirror-style webcam display.
            frame = cv2.flip(frame, 1)

            # Convert from OpenCV's BGR format to RGB for MediaPipe.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run face landmark detection on the current frame.
            results = face_mesh.process(rgb_frame)

            # Default values for the current frame.
            mouth_ratio = None
            face_detected = 0

            # ----------------------------------------
            # Process the first detected face, if available
            # ----------------------------------------
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Mark that a face was successfully found in this frame.
                    face_detected = 1

                    # Draw the face mesh on the frame for visual confirmation.
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )

                    # Compute the mouth-open ratio for this face.
                    mouth_ratio = compute_mouth_open_ratio(face_landmarks)

                    # Phase 1 logging uses only one face, so stop after the first.
                    break

            # ----------------------------------------
            # Display the current signal status on the frame
            # ----------------------------------------
            if mouth_ratio is not None:
                cv2.putText(
                    frame,
                    f"Mouth Open Ratio: {mouth_ratio:.4f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            # ----------------------------------------
            # Save a record for this frame
            # ----------------------------------------
            # If no face was detected, mouth_ratio is stored as None.
            log_data.append(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": timestamp_sec,
                    "mouth_ratio": mouth_ratio,
                    "face_detected": face_detected,
                }
            )

            # Optional terminal print for quick debugging.
            print(
                f"Frame {frame_index} | "
                f"Time: {timestamp_sec:.2f}s | "
                f"Face detected: {face_detected} | "
                f"Mouth ratio: {mouth_ratio}"
            )

            # Show the current processed frame.
            cv2.imshow("Mouth Signal Logger", frame)

            # Exit the loop when the user presses 'q'.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ----------------------------------------
    # Clean up webcam and display resources
    # ----------------------------------------
    cap.release()
    cv2.destroyAllWindows()

    # ----------------------------------------
    # Save the logged data to CSV
    # ----------------------------------------
    df = pd.DataFrame(log_data)
    df.to_csv(output_csv, index=False)

    print(f"\nSaved mouth signal log to: {output_csv}")
    print(f"Total frames logged: {len(df)}")


if __name__ == "__main__":
    main()