from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

model_path = "SoilNet_93_86.h5"

try:
    SoilNet = load_model(model_path)
except:
    SoilNet = None

classes = {
    0:"Alluvial Soil:-{ Rice,Wheat,Sugarcane,Maize,Cotton,Soyabean,Jute }",
    1:"Black Soil:-{ Virginia, Wheat , Jowar,Millets,Linseed,Castor,Sunflower}",
    2:"Clay Soil:-{ Rice,Lettuce,Chard,Broccoli,Cabbage,Snap Beans }",
    3:"Red Soil:{ Cotton,Wheat,Pulses,Millets,OilSeeds,Potatoes }"
}

def model_predict(image_path, model):
    image = load_img(image_path, target_size=(224,224))
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)

    result = np.argmax(model.predict(image))

    if result == 0:
        return "Alluvial", "Alluvial.html"
    elif result == 1:
        return "Black", "Black.html"
    elif result == 2:
        return "Clay", "Clay.html"
    elif result == 3:
        return "Red", "Red.html"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if SoilNet is None:
        return "Model not loaded"

    file = request.files['image']
    filename = secure_filename(file.filename)

    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)

    pred, output_page = model_predict(file_path, SoilNet)

    return render_template(output_page, pred_output=pred, user_image=file_path)
