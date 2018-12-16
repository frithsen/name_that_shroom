import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K 
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
import tensorflow as tf


app = Flask(__name__)

def get_model():
    global model,graph
    model = load_model('mushroom_model_5_classes_cleaner_adam.new_ratio.clean.h5')
    print(" + Model Loaded, woohoo!")
    print(model)
    print(model.summary())
    graph=tf.get_default_graph()

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image*(1/255)
    image = np.expand_dims(image, axis=0)

    return image

print(" + Loading Keras model...")
get_model()

@app.route("/predict", methods=["POST"])
def predict():
    
    message = request.get_json(force=True)
    encoded = message['image' ]
    decoded = base64.b64decode(encoded)

    with graph.as_default():
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size=(112, 112)) # was 224,224

        prediction = model.predict(processed_image).tolist()
        print(prediction)
        
        max_prob = max(prediction[0])
        print(max_prob)

        prediction_array = np.asarray(prediction)
        max_index = np.where(prediction_array==max_prob)
        
        max_index2 = max_index[1]
        print(max_index2)

        if max_prob <= .5:
            edible = 'Unsure'
            mushroom = 'Unsure'
            expensive = 'Unsure'
        elif max_index2==0:
            mushroom = 'Most Likely a Cauliflower'
            if prediction[0][1] > .25 or prediction[0][3] >.25:
                expensive = 'Chance of misclassification high - double check'
            else:
                expensive = 'No'
            if prediction[0][2]<.05:
                edible = 'Most Likely'    
            else:
                edible = 'Check with an expert first, chance of poisoning too high'
        elif max_index2==1:
            mushroom = 'Most Likely a Chanterelle'
            expensive = 'Yes'
            if prediction[0][2]<.05:
                edible = 'Most Likely'
            else:
                edible = 'Check with an expert first, chance of poisoning too high'  
        elif max_index2==2:
            expensive = 'No'
            edible = 'NO! IT IS POISONOUS! DO NOT EAT!'
            mushroom = 'Most Likely a False Morel'
        elif max_index2==3:
            mushroom = 'Most Likely a Morel'
            expensive = 'Yes'
            if prediction[0][2]<.05:
                edible = 'Most Likely but check the inside of the stem (refer to picture above)'
            else:
                edible = 'Not recommended, chance of poisoning too high. Can double check inside of the stem (refer to picture above)'    
        elif max_index2==4:
            mushroom = 'Most Likely a Porcini'
            if prediction[0][1] > .25 or prediction[0][3] >.25:
                expensive = 'Chance of misclassification high - double check'
            else:
                expensive = 'No'
            if prediction[0][2]<.05:
                edible = 'Check for crowded white gills underneath the cap, a white bulb at the bottom of the stem & a rose-like odor! Could be a Poisonous Death Cap'
            else:
                edible = 'Check with an expert first, chance of poisoning too high'
        else:
            mushroom = 'Error'
            expensive = 'Error'
            edible = 'Error'
                


    response = {
        'prediction': {
            'Cauliflower': (prediction[0][0])*100,
            'Chanterelle': (prediction[0][1])*100,
            'Falseorel': (prediction[0][2])*100,
            'Morel': (prediction[0][3])*100,
            'Porcini': (prediction[0][4])*100,
            'Edible': edible,
            'Mushroom': mushroom,
            'Expensive': expensive
        }
    }

    return jsonify(response)
    
