import os
import numpy as np
import cv2
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- Load Data ---
DATA_DIR = 'data'
X, y = [], []
gesture_labels = {'up': 0, 'left': 1, 'down': 2, 'right': 3}
label_names = {0: 'Up', 1: 'Left', 2: 'Down', 3: 'Right'}

def normalize_landmarks(landmarks):
    """Make hand landmarks relative to wrist and scale by max distance."""
    landmarks = landmarks.reshape(-1, 2)
    wrist = landmarks[0]
    landmarks -= wrist  # make wrist (0,0)
    max_val = np.max(np.linalg.norm(landmarks, axis=1))
    if max_val > 0:
        landmarks /= max_val
    return landmarks.flatten()

for gesture, label in gesture_labels.items():
    folder = os.path.join(DATA_DIR, gesture)
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            data = np.loadtxt(os.path.join(folder, file), delimiter=",")
            data = normalize_landmarks(data)
            X.append(data)
            y.append(label)

X = np.array(X)
y = np.array(y)
print(f"Loaded {len(X)} samples.")

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# --- Train Model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {acc*100:.2f}%")

# --- Live Testing ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    print("Showing live predictions. Press ESC to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        prediction_text = "No Hand"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()
                landmarks = normalize_landmarks(landmarks)
                landmarks = scaler.transform([landmarks])

                # Predict gesture
                pred = knn.predict(landmarks)[0]
                prediction_text = f"Gesture: {label_names[pred]}"

        cv2.putText(frame, prediction_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Gesture Live Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
