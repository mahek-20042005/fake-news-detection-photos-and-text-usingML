# backend/app.py

import io
import os
import hashlib
import joblib
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# ==============================================================================
# PATH SETUP & MODEL LOADING
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- Text Model ---
TEXT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'ml_text', 'text_only_model.joblib')
text_only_model = None
try:
    text_only_model = joblib.load(TEXT_MODEL_PATH)
    print("✅ Text-only ML model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Text model not found at {TEXT_MODEL_PATH}")

# --- Image Model (Local Keras Model) ---
IMAGE_MODEL_PATH = os.path.join(PROJECT_ROOT, 'ml_image', 'saved_models', 'deepfake_detector_v1two.h5')
image_model = None

try:
    image_model = load_model(IMAGE_MODEL_PATH)
    print("✅ Custom deepfake detector model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading image model: {e}")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def generate_text_hash(text: str) -> str:
    """Generates a SHA-256 hash for a given text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def generate_image_hash(image_bytes: bytes) -> str:
    """Generates a SHA-256 hash for a given image's byte content."""
    return hashlib.sha256(image_bytes).hexdigest()

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalize image for prediction."""
    image = image.resize((224, 224))        # change size if your model uses a different input shape
    arr = np.array(image) / 255.0           # normalize
    arr = np.expand_dims(arr, axis=0)       # add batch dimension
    return arr

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.route("/predict_text", methods=["POST"])
def handle_text_prediction():
    if not text_only_model:
        return jsonify({"error": "Model unavailable"}), 500

    try:
        data = request.get_json(silent=True) or {}
        text_input = (data.get("text") or "").strip()

        if not text_input:
            return jsonify({"error": "Please provide text content"}), 400

        input_data = [text_input]
        prediction_label = text_only_model.predict(input_data)[0]
        prediction_int = 1 if prediction_label == 'fake' else 0

        probabilities = text_only_model.predict_proba(input_data)[0]
        classes = list(text_only_model.classes_)
        fake_idx = classes.index('fake')
        proba_fake = float(probabilities[fake_idx])
        fake_score = int(round(proba_fake * 9))

        content_hash = generate_text_hash(text_input)

        return jsonify({
            "predicted_label": prediction_int,
            "fake_score": fake_score,
            "hash": content_hash
        })

    except Exception as e:
        print(f"Error in /predict_text: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict_image", methods=["POST"])
def handle_image_prediction():
    if not image_model:
        return jsonify({"error": "Image model unavailable"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = preprocess_image(image)

        prediction = image_model.predict(img_array)[0][0]
        predicted_label = "Fake" if prediction > 0.5 else "Real"
        prediction_int = 1 if predicted_label == "Fake" else 0

        fake_score = int(round(float(prediction) * 9))
        content_hash = generate_image_hash(image_bytes)

        return jsonify({
            "predicted_label": prediction_int,
            "fake_score": fake_score,
            "hash": content_hash
        })

    except Exception as e:
        print(f"Error in /predict_image: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "text_model_loaded": bool(text_only_model),
        "image_model_loaded": bool(image_model)
    })


if __name__ == "__main__":
    print("\nSERVER BOOT INFO")
    print(f"SCRIPT_DIR:   {SCRIPT_DIR}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"TEXT_MODEL_PATH:   {TEXT_MODEL_PATH}")
    print(f"IMAGE_MODEL_PATH:  {IMAGE_MODEL_PATH}")
    print("====================================================\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
