import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm

# Define the path to your dataset
DATASET_PATH = "./dataset/TESS"

# Emotion mapping (TESS has 7 emotions only)
emotion_map = {
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'ps': 'pleasant_surprise',
    'sad': 'sad'
}

def extract_features(path):
    y, sr = librosa.load(path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

X = []
y = []

print("Extracting features from audio files...")

for folder in tqdm(os.listdir(DATASET_PATH)):
    folder_path = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            try:
                # Fix: extract the actual emotion from the last part of filename
                label = os.path.splitext(file)[0].split("_")[-1].lower()  # 'happy'
                label = emotion_map.get(label, label)
                feature = extract_features(file_path)
                X.append(feature)
                y.append(label)
            except Exception as e:
                print(f"Error: {file_path}, {e}")

# Convert to numpy
X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save model
model_dir = "backend/model"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "emotion_model.pkl"))
print("✅ Model saved to backend/model/emotion_model.pkl")
