import os
import pandas as pd
import numpy as np
import keras

from dogDict import dog_labels_categories
from flask import Flask, request, render_template, flash, redirect, url_for, session, logging, jsonify
from keras.preprocessing import image
from keras import backend as K 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
# mongo db connection

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

model = None
graph = None

# Loading a keras model with flask
def load_model():
    global model
    global graph 
    model = keras.models.load_model("cnn25_trained.h5")
    graph = K.get_session().graph

load_model()

def prepare_image(img):
    # Convert the image to a numpy array
    img = image.img_to_array(img)
    # Scale from 0 to 255
    img = img / 255
    # Append it to test_images
    test_images = []
    test_images.append(img)
    # Convert test_images to 
    test = np.array(test_images)
    # Return the processed feature array
    return test

global filename
global prediction

@app.route('/namethatdog', methods=['GET', 'POST'])
def upload_file():
    
    data = {"success": False}
    filename = ""
    
    if request.method == 'POST':
        print(request)  

        if request.files.get('file'):
            # Read the file
            file = request.files['file']
            
            # Read the filename
            filename = file.filename
            data["filename"] = filename

            # Create a path to the Uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file to Uploads folder
            file.save(filepath)

            # Load the saved image using Keras
            # Resize it to 100x100 pixels
            image_size = (100, 100)
            im = image.load_img(filepath, target_size=image_size,
                                grayscale=False)

            # Convert the 2D image to an array of pixel values
            image_array = prepare_image(im)
            print(image_array)

            # Get the tensorflow default graph and use it to make predictions
            global graph
            with graph.as_default():

                # Use the model to make a prediction
                predicted_dog_class = model.predict_classes(image_array)[0]
                predicted_label = dog_labels_categories[predicted_dog_class]
                prediction = str(predicted_dog_class) + ": " + predicted_label
                data["prediction"] = prediction

                # Indicate that the request was a success
                data["success"] = True

            print(data)
        
            # Render another template displaying the image and its prediction
            return render_template("namethatdog.html", data=data)

    return render_template('ml.html')   

@app.route('/map')
def map():
    return render_template('dog-map.html')

@app.route('/stats')
def stats():
    return render_template('dog-stats.html')

@app.route('/overview')
def overview():
    return render_template('overview.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
