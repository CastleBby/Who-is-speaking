"""
runs the Phase 1 live webcam pipeline for the "Who is Speaking?"
project. It opens the webcam, detects a face using MediaPipe Face Mesh,
draws facial landmarks, computes a simple mouth-open ratio, and displays
that value on the video feed

USAGE: PYTHONPATH=. python scripts/run_webcam.py
"""
import cv2
import mediapipe as mp

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
        Run the live webcam face mesh and mouth signal pipeline.

    This function:
    1. initializes MediaPipe Face Mesh,
    2. opens the webcam,
    3. reads frames continuously,
    4. detects face landmarks,
    5. computes a mouth-open ratio for the detected face,
    6. displays the result on the frame,
    7. exits when the user presses 'q'.

    Returns
    -------
    None
    """
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # values come from src/config.py and help keep the video feed consistent
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=MAX_NUM_FACES,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as face_mesh:

        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            mouth_ratio = None

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )

                    mouth_ratio = compute_mouth_open_ratio(face_landmarks)
                    print(f"Mouth open ratio: {mouth_ratio:.4f}")
                    break

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

            cv2.imshow("Phase 1 - Face Mesh and Mouth Signal", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()