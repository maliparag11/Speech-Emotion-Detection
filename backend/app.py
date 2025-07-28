from flask import Flask, request, jsonify
import os
import librosa
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("backend/model/emotion_model.pkl")

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        filepath = os.path.join("temp", file.filename)
        file.save(filepath)

        features = extract_features(filepath).reshape(1, -1)
        prediction = model.predict(features)[0]

        os.remove(filepath)
        return jsonify({"emotion": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    if not os.path.exists("temp"):
        os.makedirs("temp")
    app.run(debug=True)
