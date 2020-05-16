import os

from EmoPy import FERModel
from flask import Flask, render_template, request, json, jsonify
from face_detector import FaceDetector
from PIL import Image
import tensorflow as tf
import keras

import base64
import cv2
import numpy as np
import requests
import aiohttp
import asyncio
import datetime
import time
import glob

DEBUG = False

# Can choose other target emotions from the emotion subset defined in fermodel.py in src directory. The function
# defined as `def _check_emotion_set_is_supported(self):`
target_emotions = ['calm', 'anger', 'happiness', 'surprise', 'disgust', 'fear', 'sadness']

graph = tf.get_default_graph()
model = FERModel(target_emotions, verbose=False)

loop = asyncio.get_event_loop()
app = Flask(__name__)
face_detector = FaceDetector('haarcascade_frontalface_default.xml')

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# saves frames and their detected emotion to debug folder
def debug_frame(image, emotion):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    img = Image.fromarray(image, 'RGB')
    img.save('./debug/' + emotion + '-' + st + '.png')

def get_largest_face(faces):
    largest_face = None
    largest_face_area = 0
    for face in faces:
        (column, row, width, height) = face
        if width*height > largest_face_area:
            largest_face_area = width*height
            largest_face = face
    return largest_face


@app.route('/')
def index():
    return jsonify({'emotion': 'LoL', 'faces': 42})


@app.route('/predict', methods=['POST'])
def predict():
    image_np = data_uri_to_cv2_img(request.values['image'])
    # Passing the frame to the predictor
    with graph.as_default():
        faces = face_detector.detect_faces(image_np)
        if len(faces) > 0:
            largest_face = get_largest_face(faces)
            arr_crop = image_np[largest_face[1]:largest_face[1]+largest_face[3], largest_face[0]:largest_face[0]+largest_face[2]]
            emotion = model.predict_from_ndarray(arr_crop)

            if DEBUG:
              debug_frame(image_np, emotion)

            return jsonify({'emotion': emotion, 'faces': json.dumps([largest_face])})
        else:
            return jsonify({'emotion': '', 'faces': json.dumps(faces)})


if __name__ == '__main__':
    app.run("0.0.0.0", port=5000)
