
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- Configuration ---
# Construct the absolute path to the model file to ensure it's always found
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'saved_models', 'deepfake_detector_v1two.h5')

# Define the image size the model was trained on
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

# --- Load the Model ---
# Load the trained model once when the script starts for efficiency
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Prediction Function ---
def predict_image(image_path):
    """
    Loads an image, preprocesses it, and returns a prediction from the model.
    The function returns a dictionary with the prediction and confidence score.
    """
    if model is None:
        return {"error": "Model is not loaded or failed to load."}

    try:
        # 1. Load the image from the file path, resizing it to what our model expects
        img = image.load_img(image_path, target_size=IMAGE_SIZE)

        # 2. Convert the image to a NumPy array
        img_array = image.img_to_array(img)

        # 3. Rescale the pixel values to be between 0 and 1, just like in training
        img_array /= 255.0

        # 4. Add an extra dimension because the model expects a "batch" of images
        img_array = np.expand_dims(img_array, axis=0)

        # 5. Make the prediction
        prediction = model.predict(img_array)
        score = float(prediction[0][0]) # The raw output from the sigmoid function

        # 6. Interpret the score and return a user-friendly result
        if score < 0.5:
            return {"prediction": "fake", "confidence": 1 - score}
        else:
            return {"prediction": "real", "confidence": score}

    except Exception as e:
        return {"error": f"An error occurred during prediction: {e}"}

# --- Example Usage ---
# This part runs only when you execute python predict.py directly.
# It's useful for a quick test to make sure the model works.
if __name__ == '__main__':
    # Create a dummy image file for testing if it doesn't exist
    test_image_path = 'test_image.png'
    try:
        from PIL import Image
        if not os.path.exists(test_image_path):
            print(f"Creating a dummy test image: {test_image_path}")
            dummy_array = np.uint8(np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH, 3) * 255)
            im = Image.fromarray(dummy_array)
            im.save(test_image_path)

        print(f"\n--- Testing prediction on '{test_image_path}' ---")
        result = predict_image(test_image_path)
        print(f"Prediction Result: {result}")

    except Exception as e:
        print(f"Could not run test. Error: {e}")