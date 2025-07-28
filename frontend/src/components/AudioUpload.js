import React, { useState } from "react";
import axios from "axios";
import "./AudioUpload.css";

function AudioUpload() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://localhost:5000/api/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setResult(`🎧 Detected Emotion: ${res.data.emotion}`);
    } catch (err) {
      console.error(err);
      setResult("❌ Error predicting emotion.");
    }
  };

  return (
    <div className="upload-container">
      <h2 className="title">🎙️ Speech Emotion Detector</h2>
      <form className="upload-form" onSubmit={handleSubmit}>
        <input
          className="file-input"
          type="file"
          accept=".wav"
          onChange={handleFileChange}
        />
        <button className="submit-button" type="submit">🔍 Predict Emotion</button>
      </form>
      {result && <p className="result">{result}</p>}
    </div>
  );
}

export default AudioUpload;
