from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

MODEL_DIR = r'D:\Mini_Project\fruit_analyzer_app\models'
FRUIT_CLASSIFICATION_MODEL_PATH = os.path.join(MODEL_DIR, 'fruit_classification_model.h5')
BANANA_STAGE_MODEL_PATH = os.path.join(MODEL_DIR, 'banana_stages_model.h5')
TOMATO_STAGE_MODEL_PATH = os.path.join(MODEL_DIR, 'tomato_stages_model.h5')
BANANA_SHELF_LIFE_MODEL_PATH = os.path.join(MODEL_DIR, 'banana_shelf_life_model.h5')
TOMATO_SHELF_LIFE_MODEL_PATH = os.path.join(MODEL_DIR, 'tomato_shelf_life_model.h5')

UPLOAD_FOLDER = r'D:\Mini_Project\fruit_analyzer_app\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def load_models(func):
    def wrapper(*args, **kwargs):
        if not hasattr(load_models, 'models_loaded'):
            load_models.models_loaded = True
            print("Models loaded successfully.")
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

@app.route('/')
def index():
    return render_template('home2.html')

@app.route('/upload-fruit', methods=['GET', 'POST'])
@load_models
def upload_fruit():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
        image = cv2.imread(file_path)
        processed_image = preprocess_image(image)
        fruit_classification_model = load_model(FRUIT_CLASSIFICATION_MODEL_PATH)
        fruit_prediction = fruit_classification_model.predict(processed_image)
        fruit_type = "Banana" if int(fruit_prediction) == 0 else "Tomato"
        fruit_classification_model = None
        return redirect(url_for('fruit_result', fruit_type=fruit_type, uploaded_image_name=uploaded_file.filename))
    return render_template('fruit.html')

@app.route('/upload-ripening', methods=['GET', 'POST'])
@load_models
def upload_ripening():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
        image = cv2.imread(file_path)
        processed_image = preprocess_image(image)
        fruit_classification_model = load_model(FRUIT_CLASSIFICATION_MODEL_PATH)
        fruit_prediction = fruit_classification_model.predict(processed_image)
        fruit_type = "Banana" if int(fruit_prediction) == 0 else "Tomato"
        fruit_classification_model = None
        if fruit_type == "Banana":
            stage_model = load_model(BANANA_STAGE_MODEL_PATH)
        else:
            stage_model = load_model(TOMATO_STAGE_MODEL_PATH)
        stage_prediction = stage_model.predict(processed_image)
        stage_model = None
        ripeness_stage = convert_to_stage_label(stage_prediction, fruit_type)
        return redirect(url_for('ripeness_result', ripeness_stage=ripeness_stage, uploaded_image_name=uploaded_file.filename))
    return render_template('ripeness.html')

@app.route('/upload-shelf-life', methods=['GET', 'POST'])
@load_models
def upload_shelf_life():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
        image = cv2.imread(file_path)
        processed_image = preprocess_image(image)
        fruit_classification_model = load_model(FRUIT_CLASSIFICATION_MODEL_PATH)
        fruit_prediction = fruit_classification_model.predict(processed_image)
        fruit_type = "Banana" if int(fruit_prediction) == 0 else "Tomato"
        fruit_classification_model = None
        if fruit_type == "Banana":
            shelf_life_model = load_model(BANANA_SHELF_LIFE_MODEL_PATH)
        else:
            shelf_life_model = load_model(TOMATO_SHELF_LIFE_MODEL_PATH)
        shelf_life_prediction = shelf_life_model.predict(processed_image)
        shelf_life_model = None
        shelf_life = convert_to_shelf_life_label(shelf_life_prediction, fruit_type)
        return redirect(url_for('shelf_life_result', shelf_life=shelf_life, uploaded_image_name=uploaded_file.filename))
    return render_template('shelflife.html')

@app.route('/fruit-result/<fruit_type>')
def fruit_result(fruit_type):
    uploaded_image_name = request.args.get('uploaded_image_name', '')
    uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image_name)
    return render_template('fruit_result.html', fruit_type=fruit_type, uploaded_image=uploaded_image_path)

@app.route('/ripeness-result/<ripeness_stage>')
def ripeness_result(ripeness_stage):
    uploaded_image_name = request.args.get('uploaded_image_name', '')
    uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image_name)
    return render_template('ripeness_result.html', ripeness_stage=ripeness_stage, uploaded_image=uploaded_image_path)

@app.route('/shelf-life-result/<shelf_life>')
def shelf_life_result(shelf_life):
    uploaded_image_name = request.args.get('uploaded_image_name', '')
    uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image_name)
    return render_template('shelf_life_result.html', shelf_life=shelf_life, uploaded_image=uploaded_image_path)

def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))  
    normalized_image = resized_image / 255.0
    return np.expand_dims(normalized_image, axis=0) 

def convert_to_stage_label(prediction, fruit_type):
    if fruit_type == "Banana":
        banana_stage_labels = ["Green", "Midripen", "Overripen", "Yellowish_Green"]
        stage_label = banana_stage_labels[np.argmax(prediction)]
    elif fruit_type == "Tomato":
        tomato_stage_labels = ["Overripe", "Ripe", "Unripe"]
        stage_label = tomato_stage_labels[np.argmax(prediction)]
    else:
        stage_label = "Unknown"
    return stage_label

def convert_to_shelf_life_label(prediction, fruit_type):
    if fruit_type == "Banana":
        banana_shelf_life_labels = ["1-5 days", "5-10 days", "10-15 days", "15-20 days", "Expired"]
        shelf_life_label = banana_shelf_life_labels[np.argmax(prediction)]
    elif fruit_type == "Tomato":
        tomato_shelf_life_labels = ["Expired", "1-5 days", "5-10 days", "10-15 days"]
        shelf_life_label = tomato_shelf_life_labels[np.argmax(prediction)]
    else:
        shelf_life_label = "Unknown"
    return shelf_life_label

if __name__ == '__main__':
    app.run(debug=True)
