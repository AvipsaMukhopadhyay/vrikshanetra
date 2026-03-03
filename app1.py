from flask import Flask, Blueprint, render_template, request
from markupsafe import Markup
import os

app = Flask(__name__)

##################
# Plant App Logic
##################

from plant.model import predict_image
from plant import utils

plant_bp = Blueprint('plant', __name__, template_folder='plant/templates', static_folder='plant/static')

@plant_bp.route('/', methods=['GET'])
def plant_index():
    return render_template('plant_index.html')

@plant_bp.route('/predict', methods=['GET', 'POST'])
def plant_predict():
    if request.method == 'POST':
        try:
            file = request.files['file']
            img = file.read()
            prediction = predict_image(img)
            res = Markup(utils.disease_dic[prediction])
            return render_template('plant_display.html', result=res)
        except Exception as e:
            return render_template('plant_index.html', result='Internal Server Error')
    return render_template('plant_index.html')

##################
# Soil App Logic
##################

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

soil_bp = Blueprint('soil', __name__, template_folder='soil/templates', static_folder='soil/static')

model_path = 'soil/SoilNet9386.h5'
SoilNet = load_model(model_path)
import joblib

rf_model = joblib.load("soil_rf_model.pkl")
rf_label = joblib.load("soil_label_encoder.pkl")

print("✅ Random Forest model loaded")

classes = [
    "Alluvial Soil- Rice,Wheat,Sugarcane,Maize,Cotton,Soyabean,Jute",
    "Black Soil- Virginia, Wheat, Jowar,Millets,Linseed,Castor,Sunflower",
    "Clay Soil- Rice,Lettuce,Chard,Broccoli,Cabbage,Snap Beans",
    "Red Soil- Cotton,Wheat,Pulses,Millets,OilSeeds,Potatoes"
]

type_templates = ["Alluvial.html", "Black.html", "Clay.html", "Red.html"]

def modelpredict(image_path, model):
    image = load_img(image_path, target_size=(224,224))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    result = np.argmax(model.predict(image))
    pred_label = classes[result]
    outputpage = type_templates[result]
    return pred_label, outputpage

@soil_bp.route('/', methods=['GET'])
def soil_index():
    return render_template('soil_index.html')

@soil_bp.route('/predict', methods=['GET', 'POST'])
def soil_predict():
    if request.method == 'POST':
        try:
            file = request.files['image']
            filename = file.filename
            upldir = os.path.join(app.root_path, 'soil', 'static', 'user_uploaded')

            if not os.path.exists(upldir):
                os.makedirs(upldir)

            filepath = os.path.join(upldir, filename)
            file.save(filepath)

            pred, outputpage = modelpredict(filepath, SoilNet)

            return render_template(outputpage, predoutput=pred, userimage=filepath)

        except Exception as e:
            return render_template('soil_index.html', predoutput='Internal Server Error')

    return render_template('soil_index.html')

##################
# Home Page
##################

@app.route('/')
def home():
    return render_template('index.html')


###############################
# ✅ Arduino Serial Integration
###############################
import threading
import serial
import time

def read_arduino_serial():
    try:
        arduino = serial.Serial('COM7', 9600, timeout=1)
        print("✅ Arduino Serial Connected")
    except Exception as e:
        print("❌ Could not connect to Arduino Serial:", e)
        return

    while True:
        try:
            if arduino.in_waiting > 0:
                raw_data = arduino.readline().decode("utf-8", errors="ignore").strip()

                if raw_data.startswith("Arduino:"):
                    raw_data = raw_data.replace("Arduino:","").strip()

                # Expecting format: "12.79,0.00"
                try:
                    ph_str, moist_str = raw_data.split(",")
                    ph = float(ph_str)
                    moisture = float(moist_str)

                    # ✅ Make prediction
                    pred_encoded = rf_model.predict([[ph, moisture]])[0]
                    prediction = rf_label.inverse_transform([pred_encoded])[0]

                    print(f"Arduino → pH={ph}, Moisture={moisture} → Prediction: {prediction}")

                except Exception as parse_error:
                    print("⚠ Parse error:", parse_error, "Data:", raw_data)

            time.sleep(0.05)

        except Exception as e:
            print("Serial Error:", e)
            break


##################
# Register Routes
##################

app.register_blueprint(plant_bp, url_prefix='/plant')
app.register_blueprint(soil_bp, url_prefix='/soil')

##################
# Main App Runner
##################

if __name__ == '__main__':
    # ✅ Start Arduino background reader
    serial_thread = threading.Thread(target=read_arduino_serial, daemon=True)
    serial_thread.start()

    # ✅ Start Flask app
    app.run(debug=False)
