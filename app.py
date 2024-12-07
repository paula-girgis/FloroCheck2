from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from flask_caching import Cache
import time

app = Flask(__name__)

# Setup caching for faster responses
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Load the TensorFlow Lite model
try:
    interpreter = tf.lite.Interpreter(model_path="best_model.tflite")
    interpreter.allocate_tensors()
    print("Model loaded successfully!")
except (OSError, ValueError) as e:
    print(f"Error loading model: {e}")
    interpreter = None  # If model can't be loaded, it will stay None

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

# Efficient image preprocessing with OpenCV
def preprocess_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224)) / 255.0
    img_array = np.expand_dims(img, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
@cache.cached(timeout=300, key_prefix=lambda: request.files['file'].filename)  # Cache prediction for 5 minutes
def predict():
    try:
        if 'file' not in request.files or not request.files['file']:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        # Preprocess the image
        img_array = preprocess_image(file)

        # Predict with the disease model using TensorFlow Lite
        if interpreter is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Measure the response time
        start_time = time.time()

        # Set input tensor for the interpreter
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], img_array)
        
        # Run inference
        interpreter.invoke()

        # Get the prediction results
        confidence_scores = interpreter.get_tensor(output_details[0]['index'])[0]

        # Measure the elapsed time for prediction
        response_time = round(time.time() - start_time, 2)

        max_confidence = float(np.max(confidence_scores))
        predicted_class_idx = np.argmax(confidence_scores)

        # Set a confidence threshold to filter out invalid images
        threshold = 0.68  # Adjust this based on your model's behavior
        if max_confidence < threshold:
            return jsonify({'error': 'Invalid photo. Please upload a plant leaf image.'}), 400

        # Map the predicted class index to the class name
        if predicted_class_idx in class_map:
            predicted_class_name = class_map[predicted_class_idx]
            return jsonify({
                'predicted_class': predicted_class_name,
                'confidence': max_confidence,
                'response_time': response_time  # Send response time
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
    # Enable debugging in local environment
    app.run(debug=True, host='0.0.0.0', port=8000)
