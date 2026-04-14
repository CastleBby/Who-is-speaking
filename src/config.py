# moderate resolution to keep it light but clear 
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# track one face first 
MAX_NUM_FACES = 4

# minimum confidence score to accept initial face detection 
# tune higher for less false detections 
# 0.5 = baseline threshold 
MIN_DETECTION_CONFIDENCE = 0.5

# confidence following the face across frames
# start at 0.5 baseline 
MIN_TRACKING_CONFIDENCE = 0.5

# mouth history of frames 
MOUTH_HISTORY = 15

# speaking minimum 
SPEAKING_THRESHOLD = 0.02

# match the distance 
MATCH_DISTANCE_THRESHOLD = 80

MAX_MATCH_DISTANCE = 80
