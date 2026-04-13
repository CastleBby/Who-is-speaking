# Face Tracking: Who is Speaking?
**Group:** Emily Castelan  
**Course:** MSLM640, Spring 2026  
**Date:** March 31, 2026  

---

### Brief application idea:  
Video call services like Zoom and Google Meets have preliminary features to highlight the speaker, but sometimes there are still communication issues when multiple people try to talk at once. Face tracking that monitors facial landmarks and mouth motion over time without the use of audio could provide a visual estimate of who is actively speaking, help indicate who wants to speak or is trying to speak, and increase turn-taking awareness in group calls.  

The overall goal and idea is to design a real time multi face tracking application that is designed to track participants and map mouth-movement patterns to estimate the current speaker and indicate when someone else wants to speak, or possible speaking overlap to mitigate miscommunication.  

### Tracking Method:  
**dataset:** real-time video frames from webcame or prerecorded call footage. I may collect data online as well as test with a reall video meeting/call to verify in real time.  

**process:** First faces will be detected and assigned a unique ID, the ID will be made persisitent so the system can track the face across frames and identify them in the call as speaking. Next, the mouth-related landmark must be identified and monitored throughout frames. An initial baseline of the mouth must be sampled and the participant with the strongest sustained mouth patterns indicative of speech will be labeled as the active speaker. If other participants are flagged as speaking at the same time all participants may be notified and reminded to allow the next participant to speak or address a turn-taking plan. 

**Key tools:**  
1. landmark tracking (Mediapipe Face Mesh)
will be implemented first because the application needs mouth landmarks and MediaPipe mesh seems to work well. 
2. tracking-by-detection 
will also be implemented in order to maintain IDs for all participants on a call