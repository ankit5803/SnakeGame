import os
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Paths
DATA_DIR = 'data'
MODEL_DIR = 'models'
MODEL_FILE = 'gesture_knn.pkl'
SCALER_FILE = 'scaler.pkl'
os.makedirs(MODEL_DIR, exist_ok=True)

gesture_labels = {'up': 0, 'left': 1, 'down': 2, 'right': 3}

def normalize_landmarks(landmarks):
    """Make hand landmarks relative to wrist and scale by max distance."""
    landmarks = landmarks.reshape(-1, 2)
    wrist = landmarks[0]
    landmarks -= wrist  # make wrist (0,0)
    max_val = np.max(np.linalg.norm(landmarks, axis=1))
    if max_val > 0:
        landmarks /= max_val
    return landmarks.flatten()

# --- Load and Normalize Data ---
X, y = [], []
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

# --- Scale Data ---
scaler = StandardScaler()
X = scaler.fit_transform(X)

# --- Train Model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# --- Evaluate ---
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy: {acc*100:.2f}%")

# --- Save Model and Scaler ---
joblib.dump(knn, os.path.join(MODEL_DIR, MODEL_FILE))
joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_FILE))
print(f"Model saved to {MODEL_DIR}/{MODEL_FILE}")
print(f"Scaler saved to {MODEL_DIR}/{SCALER_FILE}")
