import cv2
import mediapipe as mp
import numpy as np
from math import sqrt, atan2, degrees
import time

# Initialize MediaPipe Hands and drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam with high definition settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)  # Set width to 1920 for HD
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)  # Set height to 1080 for HD

def calculate_distance(point1, point2):
    return sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def calculate_angle(point1, point2, point3):
    angle = degrees(atan2(point3.y - point2.y, point3.x - point2.x) -
                    atan2(point1.y - point2.y, point1.x - point2.x))
    return angle + 360 if angle < 0 else angle

# Gesture recognition functions
def is_thumb_up(landmarks):
    return (landmarks[4].y < landmarks[3].y < landmarks[2].y < landmarks[1].y)

def is_thumb_down(landmarks):
    return (landmarks[4].y > landmarks[3].y > landmarks[2].y > landmarks[1].y)

def is_peace_sign(landmarks):
    return (landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and
            all(landmarks[i].y > landmarks[i - 2].y for i in [16, 20]))

def is_ok_sign(landmarks):
    thumb_index_dist = calculate_distance(landmarks[4], landmarks[8])
    return thumb_index_dist < 0.05

def is_waving(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    return calculate_distance(thumb_tip, index_tip) > 0.1

def is_fist_bump(landmarks):
    return all(landmarks[i].y > landmarks[i - 2].y for i in [8, 12, 16, 20])

def is_high_five(landmarks):
    return all(landmarks[i].y < landmarks[i - 2].y for i in [8, 12, 16, 20])

def is_clenched_fist(landmarks):
    return all(landmarks[i].y > landmarks[i - 2].y for i in [8, 12, 16, 20])

def is_pointing(landmarks):
    return landmarks[8].y < landmarks[6].y and all(landmarks[i].y > landmarks[i - 2].y for i in [12, 16, 20])

def is_crossed_fingers(landmarks):
    return landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y

def is_shaka_sign(landmarks):
    return (landmarks[4].y < landmarks[3].y < landmarks[2].y < landmarks[1].y and
            landmarks[20].y < landmarks[19].y < landmarks[18].y < landmarks[17].y and
            landmarks[12].y > landmarks[10].y and landmarks[8].y > landmarks[6].y and
            landmarks[16].y > landmarks[14].y)

# Add more gesture functions...

# Gesture functions dictionary
gesture_functions = {
    'Thumbs Up': is_thumb_up,
    'Thumbs Down': is_thumb_down,
    'Peace Sign': is_peace_sign,
    'OK Sign': is_ok_sign,
    'Waving': is_waving,
    'Fist Bump': is_fist_bump,
    'High Five': is_high_five,
    'Clenched Fist': is_clenched_fist,
    'Pointing': is_pointing,
    'Crossed Fingers': is_crossed_fingers,
    'Shaka Sign': is_shaka_sign,
    # Add remaining gesture functions here
}

# Main function to process the webcam feed
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    ptime = 0  # Initialize previous time for FPS calculation

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark

                for gesture_name, gesture_function in gesture_functions.items():
                    if gesture_function(landmarks):
                        cv2.putText(frame, gesture_name, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 139), 2)  # Dark blue color
                        break

        ctime = time.time()  # Current time
        fps = 1 / (ctime - ptime)  # FPS calculation
        ptime = ctime  # Update previous time

        # Display FPS on the image
        cv2.putText(frame, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 139), 2)  # Dark blue color
        
        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
