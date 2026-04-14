# moderate resolution to keep it light but clear 
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# track one face first 
MAX_NUM_FACES = 4

# minimum confidence score to accept initial face detection 
# tune higher for less false detections 
# 0.5 = baseline threshold 
MIN_DETECTION_CONFIDENCE = 0.4

# confidence following the face across frames
# start at 0.5 baseline 
MIN_TRACKING_CONFIDENCE = 0.4

# mouth history of frames 
MOUTH_HISTORY = 18

# speaking minimum 
SPEAKING_THRESHOLD = 0.08

# match the distance 
MATCH_DISTANCE_THRESHOLD = 80

MAX_MATCH_DISTANCE = 80

HISTORY_SIZE = 20

# to avoid tiny movements being false positives if no one is speaking
ACTIVE_SPEAKER_THRESHOLD = 0.010


SPEAKER_MARGIN = 0.004

# rules to break the one person from hogging the mic 

WANTS_TO_SPEAK_THRESHOLD = 0.008
WANTS_TO_SPEAK_RATIO = 0.65