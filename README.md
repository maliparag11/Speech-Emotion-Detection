# 🎧 Speech Emotion Detection via Audio

This project uses machine learning and deep learning techniques to detect **emotions from speech audio files**. It allows users to upload a `.wav` file and get the predicted emotional state based on the voice input.

## 🚀 Features

- 🎙️ Upload `.wav` audio files and detect emotions
- 📊 Predicts emotions like *happy, sad, angry, fearful, disgusted, surprised,* and *neutral*
- 🔗 React.js frontend with a Flask backend
- 🔁 Real-time prediction using pre-trained deep learning model
- 📁 Tested on the [TESS (Toronto Emotional Speech Set)]((https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess))

---

## 📂 Project Structure
"'Speech-Emotion-Detector/
├── backend/
│   ├── app.py
|   ├── train_model.py
│   ├── backend/
|   |   └──model/
|   |      └── emotion_model.pkl
|   ├── dataset/
|   |   └──TESS/
|   |      └──OAF_angry/
|   |         └──OAF_back_angry.wav ...
|   |      └──OAF_disgust/
|   |         └──OAF_back_disgust.wav ... 
│   └── requirements.txt
├── frontend/
|   ├── node_modules/
│   ├── public/
│   ├── src/
│   │   └── components/
│   │       └── AudioUpload.jsx
|   |       └── AudioUpload.css
|   |   └── App.js
│   └── package.json
└── README.md
"'

📸 UI Preview




🧠 How Emotion is Predicted

  1) Audio is uploaded by the user (.wav)
  
  2) Flask uses SpeechRecognition to convert audio to text
  
  3) Text is processed with NLP (TF-IDF + ML Model)
  
  4) Predicted emotion (happy, sad, etc.) is returned
  
  5) React displays the result beautifully

🧠 Model
The deep learning model is built using Keras (TensorFlow backend). It uses a CNN or LSTM architecture trained on extracted MFCC features of audio signals.

Model training file is available inside /backend/model/ or can be trained using train_model.py if provided.


📌 Sample Emotions Detected

  😄 Happy
  
  😢 Sad
  
  😠 Angry
  
  😐 Neutral

🙋‍♂️ Author
Parag Mali And Rutuja Mahajan
📧 paragmali25@gmail.com | rutujamahajan32@gmail.com
🔗 LinkedIn
