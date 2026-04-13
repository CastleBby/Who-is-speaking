# Project: "Who is speaking?"  
**Author:** Emily Castelan 
**Date:** April 12 2026 

---
## Problem: 
Video conferencing platforms such as Zoom and Google Meets have preliminary features to highlight the speaker, but sometimes there are still communication issues when multiple people try to talk at once. Face tracking that monitors facial landmarks and mouth motion over time without the use of audio could provide a visual estimate of who is actively speaking, help indicate who wants to speak or is trying to speak, and increase turn-taking awareness in group calls.  

## Pipeline Overview:
Phase 1: DETECTION - Detect mouth motion signal for one face  
Phase 2: TRACKING - Add multi-face tracking with persistent IDs  
Phase 3: FEATURES - Estimate active Speaker  
Phase 4: SPEAKER - Add overlap and turn taking cues  

## How to Run Demo: 


## File Organization:
```
who is speaking/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── sample_videos/
│   │   └── screenshots/
│   ├── processed/
│   └── annotations/
│
├── notebooks/
│   ├── 01_explore_mediapipe.ipynb
│   ├── 02_mouth_motion_features.ipynb
│   └── 03_speaker_logic_tests.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── config.py
│   ├── main.py
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── face_detector.py
│   │   └── landmark_detector.py
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── face_tracker.py
│   │   └── track_utils.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── mouth_features.py
│   │   └── temporal_features.py
│   │
│   ├── speaker/
│   │   ├── __init__.py
│   │   ├── speaker_logic.py
│   │   └── overlap_detection.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── draw_utils.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io_utils.py
│       └── video_utils.py
│
├── scripts/
│   ├── run_webcam.py
│   ├── run_video.py
│   └── test_pipeline.py
│
├── outputs/
│   ├── demo_videos/
│   ├── logs/
│   └── figures/
│
└── docs/
    ├── proposal.md
    ├── design_notes.md
    └── final_report_assets/
```