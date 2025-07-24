import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Directory to save gesture data
DATA_DIR = 'data'
GESTURES = {'w': 'up', 'a': 'left', 's': 'down', 'd': 'right'}
os.makedirs(DATA_DIR, exist_ok=True)

# Create subfolders for each gesture
for g in GESTURES.values():
    os.makedirs(os.path.join(DATA_DIR, g), exist_ok=True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    print("Press W (up), A (left), S (down), D (right) to save gesture samples. Press ESC to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror image for natural view
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract normalized landmark positions (42 values: x, y for 21 points)
                landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()

                # Capture and save gesture when key is pressed
                key = cv2.waitKey(1) & 0xFF
                if chr(key) in GESTURES:
                    gesture_name = GESTURES[chr(key)]
                    filename = os.path.join(DATA_DIR, gesture_name, f"{time.time()}.csv")
                    np.savetxt(filename, [landmarks], delimiter=",")
                    print(f"Saved sample for: {gesture_name}")

        cv2.imshow("Collect Gestures", frame)

        # Check for ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
