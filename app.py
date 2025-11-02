from flask import Flask, render_template, request, send_from_directory, url_for, redirect
import os, sqlite3, subprocess, sys
from datetime import datetime
from werkzeug.utils import secure_filename
import numpy as np

# Lazy import for TensorFlow model loading to avoid import errors during packaging
def load_model_safe(path):
    try:
        from tensorflow.keras.models import load_model as _load_model
        return _load_model(path)
    except Exception as e:
        print("Warning: could not import/load TensorFlow model:", e, file=sys.stderr)
        return None

APP = Flask(__name__)
APP.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(APP.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'face_emotionModel.h5'
# Auto-download pretrained model if missing
if not os.path.exists(MODEL_PATH):
    print('Model not found locally. Attempting to download pretrained model...')
    try:
        # try curl first
        subprocess.run(['curl', '-L', '-o', MODEL_PATH,
                        'https://github.com/justinshenk/fer-data/raw/master/fer2013_mini.h5'],
                       check=True)
        print('Model downloaded via curl.')
    except Exception as e:
        try:
            # fallback to using wget if available
            subprocess.run(['wget', '-O', MODEL_PATH,
                            'https://github.com/justinshenk/fer-data/raw/master/fer2013_mini.h5'],
                           check=True)
            print('Model downloaded via wget.')
        except Exception as e2:
            print('Automatic model download failed:', e2, file=sys.stderr)

MODEL = load_model_safe(MODEL_PATH)

# emotion labels mapping (may differ depending on model)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS submissions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT, matric TEXT, email TEXT,
                  image_path TEXT, emotion TEXT, created_at TEXT)''')
    conn.commit()
    conn.close()

init_db()

@APP.route('/')
def index():
    return render_template('index.html')

@APP.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(APP.config['UPLOAD_FOLDER'], filename)

@APP.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name', '').strip()
    matric = request.form.get('matric', '').strip()
    email = request.form.get('email', '').strip()
    file = request.files.get('photo', None)

    if not file:
        return 'No file uploaded', 400

    # secure filename and save
    filename = secure_filename(f"{matric}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}")
    filepath = os.path.join(APP.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if MODEL is None:
        emotion = 'model_not_loaded'
        message = 'Model not loaded. Please ensure face_emotionModel.h5 is present or server has internet to auto-download.'
        image_url = url_for('uploaded_file', filename=filename)
    else:
        # preprocess image for model: convert to grayscale 48x48 if possible
        try:
            from tensorflow.keras.preprocessing.image import load_img, img_to_array
            img = load_img(filepath, color_mode='grayscale', target_size=(48,48))
            arr = img_to_array(img).astype('float32') / 255.0
            arr = np.expand_dims(arr, 0)  # shape (1,48,48,1)
            preds = MODEL.predict(arr)
            pred_idx = int(np.argmax(preds))
            emotion = EMOTIONS[pred_idx] if pred_idx < len(EMOTIONS) else 'unknown'
        except Exception as e:
            emotion = 'prediction_error'
            print('Prediction error:', e, file=sys.stderr)

        # Friendly messages mapping
        friendly_map = {
            'happy': "You are smiling. You seem happy!",
            'sad': "You are frowning. Why are you sad?",
            'angry': "You look angry. Take a deep breath — breathe in, breathe out.",
            'surprise': "You look surprised! What happened?",
            'fear': "You seem worried or afraid. Are you okay?",
            'disgust': "You look displeased. Is something bothering you?",
            'neutral': "You look calm and neutral."
        }
        message = friendly_map.get(emotion, f'Emotion detected: {emotion}')
        image_url = url_for('uploaded_file', filename=filename)

    # Save submission to DB
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('INSERT INTO submissions (name, matric, email, image_path, emotion, created_at) VALUES (?, ?, ?, ?, ?, ?)',
              (name, matric, email, filepath, emotion, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

    # Return simple HTML result
    html = f"""<!doctype html>
    <html><head><meta charset='utf-8'><title>Result</title></head>
    <body>
    <h2>{message}</h2>
    <p><strong>Name:</strong> {name} &nbsp; <strong>Matric:</strong> {matric}</p>
    <img src='{image_url}' style='max-width:400px;' alt='uploaded image'/>
    <p><a href='/'>Go back</a></p>
    </body></html>"""
    return html

if __name__ == '__main__':
    APP.run(host='0.0.0.0', port=5000, debug=True)
