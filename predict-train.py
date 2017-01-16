"""
Model
"""
import os
import argparse
import csv
import math
import numpy as np
import cv2
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Input, Activation, Dropout, Conv2D, Flatten, MaxPooling2D, Convolution2D
from keras.layers.advanced_activations import ELU
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import socketio
import eventlet
import eventlet.wsgi
import time
import base64
import json
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from model import *

'''
Globals 
'''
model = None
dst_model_file = None
grayscale=False
imgCache=[]
steeringCache=[]
sio = socketio.Server()
app = Flask(__name__)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    
@sio.on('predict')
def predict(sid, imgString):
    global model
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    normImg = normalizeImage(image_array)
    steering_angle = float(model.predict(normImg, batch_size=1))
    return steering_angle

@sio.on("update")
def update(sid, imgString, steering):
    global model, imgCache, steeringCache
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image = np.asarray(image)
    normalizedImage = normalizeImage(image)
    normalizedImage = normalizedImage.reshape(normalizedImage.shape[1:])
    
    if len(steeringCache) >= 10:
        print('training')
        X = np.array(imgCache, dtype=np.float32)
        y = np.array(steeringCache, dtype=np.float32)
        imgCache=[]
        steeringCache=[]
        try:
            history = model.fit(X, y, batch_size=y.shape[0], nb_epoch=1, verbose=2)
            model.save(dst_model_file)
            print('Done')
        except:
            e = sys.exc_info()[0]
            print( "Error: %s" % e )

    else:
        imgCache.append(normalizedImage)
        steeringCache.append(steering)
    print('cache len: {0}'.format(len(steeringCache)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Improve a model')
    parser.add_argument('-m', dest='model', help='model file',required=True)
    parser.add_argument('-i', dest='input', help='input model number', type=int, default=-1)
    parser.add_argument('-o', dest='output', help='output model number', type=int,required=True)
    parser.add_argument('-g', dest='gray', help='grayscale', action='store_true')
    args = parser.parse_args()
    
    if args.input == -1:
        src_model_file = args.model
    else:
        src_model_file = args.model.replace('.h5', '.{0}.h5'.format(args.input))

    grayscale = args.gray

    dst_model_file = args.model.replace('.h5', '.{0}.h5'.format(args.output))
    model = load_model(src_model_file)
    
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 5678)), app)
