from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np


app = Flask(__name__)

# Load the model with error handling
try:
    model = tf.keras.models.load_model(r"best_model.keras")
    print("Model loaded successfully!")
except (OSError, ValueError) as e:
    print(f"Error loading model: {e}")
    model = None  # Set to None to prevent usage if not loaded

# Define the disease classes
class_indices = {
    'Apple Cedar Rust': 0,
    'Apple Healthy': 1,
    'Apple Scab': 2,
    'Bluberry Healthy': 3,
    'Citrus Black Spot': 4,
    'Citrus Canker': 5,
    'Citrus Healthy': 6,
    'Corn Gray Leaf Spot': 7,
    'Corn Northern Leaf Blight': 8,
    'Grape Healthy': 9,
    'Pepper,bell Bacterial Spot': 10,
    'Pepper,bell Healthy': 11,
    'Potato Early Blight': 12,
    'Potato Healthy': 13,
    'Potato Late Blight': 14,
    'Raspberry Healthy': 15,
    'Strawberry Healthy': 16,
    'Strawberry Leaf Scorch': 17,
    'Tomato Early Blight': 18,
    'Tomato Healthy': 19,
    'Tomato Late Blight': 20
}

class_map = {value: key for key, value in class_indices.items()}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files or not request.files['file']:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        try:
            img = Image.open(file)
        except UnidentifiedImageError:
            return jsonify({'error': 'Invalid image file. Ensure the file is a valid image.'}), 400

        # Resize and preprocess the image
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict with the disease model
        prediction = model.predict(img_array)
        confidence_scores = prediction[0]
        max_confidence = float(np.max(confidence_scores))
        predicted_class_idx = np.argmax(confidence_scores)
        

        # # Set a confidence threshold to filter out invalid images
        threshold = 0.68  # Adjust this based on your model's behavior
        
        if max_confidence < threshold:
            return jsonify({'error': 'Invalid photo. Please upload a plant leaf image.'}), 400

        # Map the predicted class index to the class name
        if predicted_class_idx in class_map:
            predicted_class_name = class_map[predicted_class_idx]
            return jsonify({
                'predicted_class': predicted_class_name,
                'confidence': max_confidence
            })
        else:
            return jsonify({'error': 'Disease not supported yet'}), 400

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {e}'}), 500


# Error handling for 404
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not Found. The API endpoint you are trying to access does not exist.'}), 404

if __name__ == '__main__':
    app.run()
