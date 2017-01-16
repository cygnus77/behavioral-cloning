"""
Model
"""
import os
import csv
import math
import argparse
import numpy as np
import cv2
import json
import tensorflow as tf
from keras.layers.core import Lambda
from keras.models import Sequential, load_model
from keras.layers import Dense, Input, Activation, Dropout, Conv2D, Flatten, MaxPooling2D, Convolution2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

DATA_DIR = 'data/'
LABELS_FILENAME = os.path.join(DATA_DIR,'driving_log.csv')

"""
Functions for training
"""
# generate training data in an infinite loop - passed into keras fit_generator
def genTrainingData(data, gray):
    while True:
        data = shuffle(data)
        src = genNormalizedData(genAugmentedViews(data), gray)
        for sample in src:
            yield sample

def genValidationData(data):
    while True:
        for row in data:
            yield (readImage(row['center']), float(row['steering']))

# enumerate rows of csv data - returns raw row contents
def enumDrivingLog(driving_log_fname):
    with open(driving_log_fname, 'r') as labels_file:
        rdr=csv.DictReader(labels_file)
        for row in rdr:
            yield row

# filter out some rows (based on keepPercentage) with low steering values (based on threshold)
def filterDrivingLog(src, threshold, keepPercentage):
    for row in src:
        steering = float(row['steering'])
        if abs(steering) > threshold or np.random.randint(0,100) <= keepPercentage:
            yield row

def genShadowsOnView(src):
    for img, steering in src:
        # add random shadows to center, right and left cameras
        yield (addShadow(img), steering)

def genDimmerViews(src):
    for img, steering in src:
        # vary image brightness on each camera - to half and third of the original brightness
        yield (adjustBrightness(img, .75), steering)
        yield (adjustBrightness(img, 1.25), steering)

def genShiftedViews(src):
    for img, steering in src:
        # add random shifts
        yield addShift(img, steering, axis=0)
        yield addShift(img, steering, axis=1)
        # flip center camera image
        yield (cv2.flip(img, 1), steering * -1.)
            
# generate images that supplement the data set
def genAugmentedViews(src):
    for row in src:
        steering = float(row['steering'])
        center = readImage(row['center'])
        left = readImage(row['left'])
        right = readImage(row['right'])
        
        imgSrcArr = []
        # center camera image
        imgSrcArr.append( (center, steering) )

        # adjust steering for left and right cameras
        left_steer = np.clip(steering + 0.12, -1, 1)
        right_steer = np.clip(steering - 0.12, -1, 1)
        imgSrcArr.append( (left, left_steer) )
        imgSrcArr.append( (right, right_steer) )

        for item in imgSrcArr:
            yield item
        for txItem in genShiftedViews(imgSrcArr):
            yield txItem
        for txItem in genDimmerViews(imgSrcArr):
            yield txItem
        for txItem in genShadowsOnView(imgSrcArr):
            yield txItem
        for txItem in genShadowsOnView(genShiftedViews(imgSrcArr)):
            yield txItem
        

# shift image horizontally by a random number of pixels and proportionally adjust steering angle
# axis 1 for horizontal, 0 for vertical
def addShift(img, steering, axis):
     
    # pick random shift amount between -1/5th width to +1/5th width
    amt = int(np.random.randint(-img.shape[axis]//5,img.shape[axis]//5+1))
    # destination image that will contain shifted image
    sImg = np.zeros(img.shape,dtype=np.uint8)
    if axis == 1:
        # compute change in steering
        steer_inc = -amt / img.shape[axis]
        #print("Steering: %.3f, Inc: %.3f" %(steering, steer_inc))
        steering += steer_inc
        # shift image by copying part of image into destination
        if amt < 0:
            amt = abs(amt)
            sImg[:,amt:img.shape[axis]] = img[:,0:img.shape[axis]-amt]
        else:
            sImg[:,0:img.shape[axis]-amt] = img[:,amt:img.shape[axis]]
    else:
        # shift image by copying part of image into destination
        if amt < 0:
            amt = abs(amt)
            sImg[amt:img.shape[axis],:] = img[0:img.shape[axis]-amt,:]
        else:
            sImg[0:img.shape[axis]-amt,:] = img[amt:img.shape[axis],:]
    # return new image and steering pair
    return sImg,steering

# adds a random shadow to a given image
def addShadow(img):
    # convert image to YUV space to get the luma (brightness) channel
    y,u,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb))
    y = y.astype(np.int32)
    # create mask image with same shape as input image
    mask = np.zeros(y.shape, dtype=np.int32)
    # compute a random line in slope, intercept form
    # random x1,x2 values (y1=0, y2=height)
    x1 = np.random.uniform() * y.shape[1]
    x2 = np.random.uniform() * y.shape[1]
    slope = float(y.shape[0]) / (x2 - x1)
    intercept = -(slope * x1)
    # assign pixels of mask below line
    for j in range(mask.shape[0]):
        for i in range(mask.shape[1]):
            if j > (i*slope)+intercept:
                mask[j,i] -= np.random.randint(30,180)
    # apply mask
    y += mask
    # ensure values are within uint8 range to avoid artifacts
    y = np.clip(y, 0,255).astype(np.uint8)
    # convert back to RGB
    return cv2.cvtColor(cv2.merge((y,u,v)), cv2.COLOR_YCrCb2RGB)

# adjust brightness of given image (img) by multiplyling V (brightness)
def adjustBrightness(img, m):
    h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    v = np.clip(v * m, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h,s,v)), cv2.COLOR_HSV2RGB)

# normalize image and steering data
# round out steering values to fall in 0.01 buckets (so problem is simplified, could even become classification problem)
# keras requires np.array for y values: convert steering to 1D np.array
def genNormalizedData(src, gray):
    for img,steering in src:
        # round steering values
        normSteering = round(steering,2)
        x = normalizeImage(img, gray)
        y = np.array(normSteering, ndmin=1, dtype=np.float)
        yield x,y

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
# convert to y,u,v or gray (keep only y)
# crop to remove top 5th and bottom 5th of image
# keras requires 4d-tensor input: add dimension 1, h,w,d for image data
def normalizeImage(img, gray):
    croppedImg = cropImage(img)
    resizedImg = cv2.resize(croppedImg, (200,66), interpolation = cv2.INTER_CUBIC)
    if gray:
        yuvImg = convColorSpace2Gray(resizedImg)
    else:
        yuvImg = convColorSpace2YUV(resizedImg)
    x = yuvImg.reshape(1,*yuvImg.shape).astype(np.float32)
    return x

# de-normalize image data to RGB format for display
def imgForDisplay(img):
    img = img.astype(np.uint8)
    img = img.reshape(img.shape[1:4])
    if img.shape[2] == 3: #yuv
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    elif img.shape[2] == 1: #grayscale
        img = img.reshape(img.shape[0:2])
    return img

# Model based on NVIDIA model
# Dropouts are not specified in the paper
def nvidia_model(input_shape, use_dropout = True):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1.,input_shape=input_shape))
    
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='same', init='he_normal'))
    model.add(ELU())
    if use_dropout:
        model.add(Dropout(0.1))
    
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='same', init='he_normal'))
    model.add(ELU())
    if use_dropout:
        model.add(Dropout(0.25))
    
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='same', init='he_normal'))
    model.add(ELU())
    if use_dropout:
        model.add(Dropout(0.25))
    
    model.add(Convolution2D(64, 3, 3, subsample=(2,2), border_mode='same', init='he_normal'))
    model.add(ELU())
    if use_dropout:
        model.add(Dropout(0.5))
    
    model.add(Convolution2D(64, 3, 3, subsample=(2,2), border_mode='same', init='he_normal'))
    model.add(ELU())
    if use_dropout:
        model.add(Dropout(0.5))
    
    model.add(Flatten())
    
    model.add(Dense(100, init='he_normal'))
    model.add(ELU())
    if use_dropout:
        model.add(Dropout(0.5))
    
    model.add(Dense(50, init='he_normal'))
    model.add(ELU())
    if use_dropout:
        model.add(Dropout(0.25))
    
    model.add(Dense(10, init='he_normal'))
    model.add(ELU())
    
    model.add(Dense(1, init='he_normal'))
    
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train NVIDIA Model on Udacity data')
    parser.add_argument('model', help='file name where model is stored. Extension must be .h5, will be added if not provided')
    parser.add_argument('-g', dest='gray', help='Convert images to grayscale', action='store_true')
    parser.add_argument('-v', dest='validation', help='Separate data into train and validation', action='store_true')
    parser.add_argument('-d', dest='dropout', help='Use dropouts', action='store_true')
    parser.add_argument('-p', dest='plot', help='Display augmented samples', action='store_true')
    args = parser.parse_args()
    
    # Get file path to model
    model_file = os.path.join(DATA_DIR, args.model)
    if not model_file.endswith('.h5'):
        model_file += '.h5'
    
    # Parse csv file, select rows for training
    data = list(filterDrivingLog(enumDrivingLog(LABELS_FILENAME), 0.01, 25))

    if args.validation:
        # split into train and validation
        train, val = train_test_split(shuffle(data), test_size=0.20)
    else:
        # use all the data for final training
        train = shuffle(data)
        val = None

    # select a row that has non-zero steering angle to be used for visualization and computing
    # image dimensions and samples per image created by augmentation
    sampleIdx = 0
    while float(train[sampleIdx]['steering']) == 0.:
        sampleIdx += 1

    # wrap selected row with data augmentation generators
    imageGenerator = genNormalizedData(genAugmentedViews([train[sampleIdx]]), args.gray)
    imgListExample = list(imageGenerator)
    numImagesPerSample = len(imgListExample)
    numRows = len(train) * numImagesPerSample
    exampleImg = imgListExample[0][0]

    # get image dimensions
    image_shape = exampleImg.shape
    print("NumSamples: {0}, Shape:{1}".format(numRows, image_shape))

    if args.plot:
        # Visualization of selected - to ensure augmented images are correct
        for normImg,normSteering in imgListExample:
            plt.figure()
            plt.axis('off')
            img = imgForDisplay(normImg)
            ht = img.shape[0]+10
            steering = normSteering # * 2.
            if len(img.shape) == 3: # rgb
                plt.imshow(img)
            else: # grayscale
                plt.imshow(img, cmap='gray')
            plt.text(0,ht,'Steering: %.3f' % steering)

    # Load existing model & weights from disk or create a new model
    if os.path.exists(model_file):
        print("Loading from file")
        model = load_model(model_file)
    else:
        print("Creating model")
        model = nvidia_model(image_shape[1:4], args.dropout)

    # print model summary
    model.summary()

    # Train
    optimizer = Adam(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    if val == None:
        history = model.fit_generator(genTrainingData(train, args.gray), samples_per_epoch=numRows, 
                      nb_epoch=25, verbose=1, max_q_size=256, pickle_safe=True, nb_worker=32)
    else:
        history = model.fit_generator(genTrainingData(train, args.gray), samples_per_epoch=numRows, 
                      nb_epoch=25, verbose=1, max_q_size=256, pickle_safe=True, nb_worker=32,
                      validation_data=genNormalizedData(genValidationData(val), args.gray), nb_val_samples=len(val))

    # Save weights and model json file
    model.save(model_file)
    jsonStr = model.to_json()
    with open(model_file.replace('h5','json'), 'w') as jf:
        jf.write(jsonStr)

    # Save learning rate history to file
    with open(model_file.replace('.h5','.history.csv'), 'w') as hist_csv:
        hist_wr = csv.writer(hist_csv)
        hist_wr.writerow(history.history['loss'])
        if args.validation:
            hist_wr.writerow(history.history['val_loss'])
