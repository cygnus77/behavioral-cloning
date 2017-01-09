"""
Model
"""
import os
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

sio = socketio.Server()
app = Flask(__name__)

DATA_DIR = 'data/'
LABELS_FILENAME = os.path.join(DATA_DIR,'driving_log.csv')
SRC_MODEL_FILE = 'data/model.good.5.h5'
DEST_MODEL_FILE = 'data/model.out.{0}.h5'.format(time.strftime("%Y%m%d.%H%M"))
model = load_model(SRC_MODEL_FILE)


"""
Operations done for every image - both training and predictions
"""
# retrive image data from disk
# returns image in RGB format
def readImage(img_fname):
    img_path = os.path.join(DATA_DIR,img_fname.strip())
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# convert RGB to YUV
def convColorSpace2YUV(img):
    y,u,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb))
    return cv2.merge((y,u,v))

# convert RGB to gray (keep 3rd dim as 1)
def convColorSpace2Gray(img):
    y,u,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb))
    return y.reshape(*y.shape,1)

# crop 5th off top and 25px off bottom of given image
def cropImage(img):
    ht = img.shape[0]
    cropHt = ht // 5
    return img[cropHt:ht-25, :]

# normalize image data
# convert to y,u,v
# crop to remove top 5th and bottom 5th of image
# adjust pixel values to fall within -1 to +1
# keras requires 4d-tensor input: add dimension 1, h,w,d for image data
def normalizeImage(img):
    croppedImg = cropImage(img)
    resizedImg = cv2.resize(croppedImg, (200,66), interpolation = cv2.INTER_CUBIC)
    yuvImg = convColorSpace2YUV(resizedImg)
    nImg = ((yuvImg / 127.5) - 1.).astype(np.float32)
    x = nImg.reshape(1,*nImg.shape)
    return x

# de-normalize image data to RGB format for display
def imgForDisplay(img):
    img = ((img + 1.) * 127.5).astype(np.uint8)
    img = img.reshape(img.shape[1:4])
    if img.shape[2] == 3: #yuv
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    elif img.shape[2] == 1: #grayscale
        img = img.reshape(img.shape[0:2])
    return img

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    
@sio.on('predict')
def predict(sid, imgString):
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    normImg = normalizeImage(image_array)
    steering_angle = float(model.predict(normImg, batch_size=1))
    return steering_angle

imgCache=[]
steeringCache=[]

@sio.on("update")
def update(sid, imgString, steering):
    global imgCache, steeringCache
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
            model.save(DEST_MODEL_FILE)
            print('Done')
        except:
            e = sys.exc_info()[0]
            print( "Error: %s" % e )
        #jsonStr = model.to_json()
        #with open(MODEL_FILE.replace('h5','json'), 'w') as jf:
        #    jf.write(jsonStr)
    else:
        imgCache.append(normalizedImage)
        steeringCache.append(steering)
    print('cache len: {0}'.format(len(steeringCache)))

if __name__ == '__main__':
    
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 5678)), app)
