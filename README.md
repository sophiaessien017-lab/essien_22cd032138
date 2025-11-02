# Emotion Detection Web App - essien_22cd032138
Simple Flask app that detects facial emotions from uploaded photos.

How to run locally:
  1. pip install -r requirements.txt
  2. python app.py
  3. Open http://127.0.0.1:5000

Notes:
- On first run the app will attempt to auto-download a small pretrained model (if internet is available).
- If auto-download fails, download a compatible model and save it as face_emotionModel.h5 in the project root.
