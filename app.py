from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

MODEL_DIR = r'D:\Mini_Project\fruit_analyzer_app\models'
BANANA_STAGE_MODEL_PATH = os.path.join(MODEL_DIR, 'banana_stages_model.h5')
TOMATO_STAGE_MODEL_PATH = os.path.join(MODEL_DIR, 'tomato_stages_model.h5')
BANANA_SHELF_LIFE_MODEL_PATH = os.path.join(MODEL_DIR, 'banana_shelf_life_model.h5')
TOMATO_SHELF_LIFE_MODEL_PATH = os.path.join(MODEL_DIR, 'tomato_shelf_life_model.h5')

UPLOAD_FOLDER = r'D:\Mini_Project\fruit_analyzer_app\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_models(func):
    def wrapper(*args, **kwargs):
        # Load the models before the first request
        if not hasattr(load_models, 'models_loaded'):
            load_models.models_loaded = True
            print("Models loaded successfully.")
        return func(*args, **kwargs)
    # Preserve the original function name
    wrapper.__name__ = func.__name__
    return wrapper

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@load_models
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
        image = cv2.imread(file_path)
        processed_image = preprocess_image(image)
        fruit_classification_model = load_model(os.path.join(MODEL_DIR, 'fruit_classification_model.h5'))
        fruit_prediction = fruit_classification_model.predict(processed_image)
        fruit_type = "Banana" if int(fruit_prediction) == 0 else "Tomato"
        fruit_classification_model = None  # Deleting the fruit classification model
       
        if fruit_type == "Banana":
            banana_stage_model = load_model(BANANA_STAGE_MODEL_PATH)
            banana_shelf_life_model = load_model(BANANA_SHELF_LIFE_MODEL_PATH)
            stage_prediction = banana_stage_model.predict(processed_image)
            shelf_life_prediction = banana_shelf_life_model.predict(processed_image)
            banana_stage_model = None  # Deleting the banana stage model
            banana_shelf_life_model = None  # Deleting the banana shelf life model
        else:
            tomato_stage_model = load_model(TOMATO_STAGE_MODEL_PATH)
            tomato_shelf_life_model = load_model(TOMATO_SHELF_LIFE_MODEL_PATH)
            stage_prediction = tomato_stage_model.predict(processed_image)
            shelf_life_prediction = tomato_shelf_life_model.predict(processed_image)
            tomato_stage_model = None  # Deleting the tomato stage model
            tomato_shelf_life_model = None  # Deleting the tomato shelf life model

        stage_prediction_label = convert_to_stage_label(stage_prediction, fruit_type)
        shelf_life_prediction_label = convert_to_shelf_life_label(shelf_life_prediction, fruit_type)


        #return render_template('result.html', fruit_type=fruit_type, stage_prediction=stage_prediction_label, shelf_life_prediction=shelf_life_prediction_label)
    
        return render_template('result.html', fruit_type=fruit_type, stage_prediction=stage_prediction_label, shelf_life_prediction=shelf_life_prediction_label, uploaded_image_name=uploaded_file.filename)

def convert_to_stage_label(prediction, fruit_type):
    banana_stage_labels = ["Green", "Midripen", "Overripen", "Yellowish_Green"]
    tomato_stage_labels = ["Overripe", "Ripe", "Unripe"]

    if fruit_type == "Banana":
        stage_label = banana_stage_labels[np.argmax(prediction)]
    elif fruit_type == "Tomato":
        stage_label = tomato_stage_labels[np.argmax(prediction)]
    else:
        stage_label = "Unknown"

    return stage_label

def convert_to_shelf_life_label(prediction, fruit_type):
    banana_shelf_life_labels = ["1-5 days", "5-10 days", "10-15 days", "15-20 days", "Expired"]
    tomato_shelf_life_labels = ["Expired", "1-5 days", "5-10 days", "10-15 days"]

    if fruit_type == "Banana":
        shelf_life_label = banana_shelf_life_labels[np.argmax(prediction)]
    elif fruit_type == "Tomato":
        shelf_life_label = tomato_shelf_life_labels[np.argmax(prediction)]
    else:
        shelf_life_label = "Unknown"

    return shelf_life_label






def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))  
    normalized_image = resized_image / 255.0
    return np.expand_dims(normalized_image, axis=0) 

if __name__ == '__main__':
    app.run(debug=True)
