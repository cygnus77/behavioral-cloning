# Behavioral Cloning


## Model Architecture

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 1)    0           lambda_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 33, 100, 24)   624         lambda_1[0][0]
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 33, 100, 24)   0           convolution2d_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 33, 100, 24)   0           elu_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 50, 36)    21636       dropout_1[0][0]
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 17, 50, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 17, 50, 36)    0           elu_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 9, 25, 48)     43248       dropout_2[0][0]
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 9, 25, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 9, 25, 48)     0           elu_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 13, 64)     27712       dropout_3[0][0]
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 5, 13, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 5, 13, 64)     0           elu_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 7, 64)      36928       dropout_4[0][0]
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 3, 7, 64)      0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 3, 7, 64)      0           elu_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1344)          0           dropout_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           134500      flatten_1[0][0]
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 100)           0           dense_1[0][0]
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 100)           0           elu_6[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_6[0][0]
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 50)            0           dense_2[0][0]
____________________________________________________________________________________________________
dropout_7 (Dropout)              (None, 50)            0           elu_7[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_7[0][0]
____________________________________________________________________________________________________
elu_8 (ELU)                      (None, 10)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          elu_8[0][0]
====================================================================================================
Total params: 270,219
Trainable params: 270,219
Non-trainable params: 0
____________________________________________________________________________________________________




## Normalization
* Input image is assumed to be RGB
* Crop 1/5th off top and 25px off bottom
* Image converted to YCbCr color space
* Scaled down to 200x66
* Pixel values converted from uint8 (0 - 255) range to float32 (-1.0 to +1.0) range


## Training data pre-processing
* Majority of the training data contains steering angles that are close to 0. To reduce the disparity, 75% of the rows with low steering values are removed.
* Round steering values to 2 decimal places. Each steering angle is one of 200 values in range -1 to +1 (steps of 0.01)
So problem is simplified and could even be treated as a classification problem with 200 classes.
* Training data is shuffled after each epoch
* 


### Data Augmentation

 NumSamples: 74992, Shape:(1, 66, 200, 3)

Center camera image:

![png](readme/output_1_2.png)

<hr>

Left camera image:

![png](readme/output_1_3.png)

<hr>

Right camera image:

![png](readme/output_1_4.png)

<hr>

Random shadow added to center image:

![png](readme/output_1_5.png)

<hr>

Random shadow added to left image:

![png](readme/output_1_6.png)

<hr>

Random shadow added to right image:

![png](readme/output_1_7.png)

<hr>
Random horizontal shift added to center image:

![png](readme/output_1_8.png)

<hr>

Random horizontal shift added to left image:

![png](readme/output_1_9.png)

Random horizontal shift added to left image:

![png](readme/output_1_10.png)

<hr>

Image flipped horizontally:

![png](readme/output_1_11.png)

<hr>

Center image dimmed:

![png](readme/output_1_12.png)

<hr>

Center image brightened:

![png](readme/output_1_13.png)

<hr>

Left image dimmed:

![png](readme/output_1_14.png)

<hr>

Left image brightened:

![png](readme/output_1_15.png)

<hr>

Right image dimmed:

![png](readme/output_1_16.png)

<hr>

Right image brightened:

![png](readme/output_1_17.png)

<hr>

## Live Trainer








