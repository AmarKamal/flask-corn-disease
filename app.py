

from flask import Flask, render_template, request, redirect, url_for, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import base64

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import time

app = Flask(__name__)

# Load the models
health_model = load_model('predictHealth.h5')
disease_model = load_model('predictDisease.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # Adjust size as per the model's expected size
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and classification
@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file
        img_path = os.path.join('static/images', file.filename)
        file.save(img_path)

        # Classify if the leaf is healthy or unhealthy
        health_img = preprocess_image(img_path)
        health_prediction = health_model.predict(health_img)

        # If healthy, show an alert and redirect to the disease classification page
        if health_prediction[0][0] > 0.5:  # Example threshold
            return jsonify({'status': 'healthy'}) 
        else:
            return jsonify({'status': 'unhealthy'})

# Route to display the disease classification page
@app.route('/disease')
def disease():
    img_path = request.args.get('img_path', None)
    if img_path:
    # Classify the disease using the second model (predictDisease.h5)
        disease_img = preprocess_image(img_path)
        #disease_img = preprocess_image('static/images/uploaded_image.jpg')
        disease_prediction = disease_model.predict(disease_img)

        # Get the class with the highest probability
        disease_class = np.argmax(disease_prediction)
        classes = ['Blight', 'Common Rust', 'Gray Leaf']

        return render_template('disease.html', result = classes[disease_class],img_path=img_path)
    return redirect(url_for('index'))
if __name__ == '__main__':
    app.run(debug=True)
