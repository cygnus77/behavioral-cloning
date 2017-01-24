# Behavioral Cloning


## Model Architecture

![png](model.png)

| Layer and Type | Output Size | Params|
|----------------|-------------|-------|
|Normalization Layer| 200x66x3 | |
|Convolution 1 | 200x66x3 | 1824 |
|ELU | | |
|Dropout| | 0.25|
|Convolution 2| 50, 17, 36 | 21636 |
|ELU | | |
|Dropout| | 0.25|
|Convolution 3| 23, 7, 48 | 43248 |
|ELU | | |
|Dropout| | 0.5|
|Convolution 4| 21, 5, 64 | 27712 |
|ELU | | |
|Dropout| | 0.5|
|Convolution 5| 19, 3, 64 | 36928 |
|ELU | | |
|Dropout| | 0.5|
|Flatten | 3648| |
|Fully Connected 1| 100 | 364900 |
|ELU | | |
|Dropout| | 0.5|
|Fully Connected 2| 50 | 5050 |
|ELU | | |
|Dropout| | 0.5|
|Fully Connected 3| 10 | 510 |
|ELU | | |
|Fully Connected 4| 1 | 11 |

I tried a few different architectures before settling on the NVIDIA model detailed in the [Apr.2016 paper titled "End to End Learning for Self-Driving Cars"](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

After implementing a model that matched NVIDIA team's description (layers and number of trainable parameters), I experimented with (i) color spaces, (ii) image augmentation, (iii) adjusting model and training parameters, (iv) normalization and (v) live-training to try and achieve these two goals:
* Car should sucessfully navigate track 1.
* Car should drive with stability and not sway from side to side.


## Normalization
* The following transforms are peformed on training, validation data sets and test data from the simulator.
#### Reading Images
* Input image read from disk are always converted to RGB format.

#### Cropping
* Crop 1/5th off top and 25px off bottom to eliminate most of the sky and car's hood. This lets the model train on what is most  important - the road!

#### Color space
* Image converted to YCbCr color space where Y is the grayscale copy of the image and Cb,Cr contain the color information.
* From tests, it became clear that training on YCbCr did not have a significant benefit over RGB.
* Keeping the image in YCbCr space made it easier to experiment with training on the Y channel alone.

#### Scaling
* Images are scaled down to 200x66 to reduce the overall size of the model.
* The exact numbers (200x66) were chosen to match the NVIDIA model.

#### Pixel value normalization
* Pixel values are converted from uint8 (0 - 255) range to float32 (-1.0 to +1.0) range.
* This step is implemented in the model as the first layer so we can take advantage of faster execution on the GPU.


## Training data pre-processing

#### Improving training data
* Majority of the training data contains steering angles that are close to 0. To reduce the disparity, 75% of the rows with low steering values are removed.

#### Steering value normalization
* Round steering values to 2 decimal places.
* Each steering angle is one of 200 values in range -1 to +1 (steps of 0.01).
* This simplifies the problem and could even be treated as a classification problem with 200 classes.


#### Data Augmentation

###### Using left and right camera images
* Every training data sample contains images from 3 camera: left, center and right along with a steering value (y)
* The left and right images can be used as center images by adjusting the steering values: adding adjustment for the left image and subtracting the adjustment value for the right image.
* I experimented with several values for the steering adustment between 0.25 and 0.04.
* Models trained with a high adjustment value tended to turn sharply causing the car to zig-zag even on a straight stretch of road.
* Models training with a low adjustment value (like .04) resulted in models that could not turn fast enough on sharp curves.
* Through trial and error, I found that 0.1 was a good value to use.
* All subsequent augmentations are performed on all three images.
* Another simple method to generate training data is to horizontally flip each image and negating the steering value. This would yield and additional three images to train on.

|Left camera image | Center camera image | Right camera image |
|------------------|---------------------|--------------------|
|![png](output_2_3.png)|![png](output_2_2.png)|![png](output_2_4.png)|

<hr>

###### Random image shifting
* Another method to generate additional data is to randomly shift each image (under 1/5 th of the width) and proportionally adjust the steering value.


|Source | Shift Vertical | Shift Horizontal | Horizontal Flip |
|-------|----------------|------------------|-----------------|
|Center camera|![png](output_2_5.png)|![png](output_2_6.png)|![png](output_2_7.png)|
|Left camera|![png](output_2_8.png)|![png](output_2_9.png)|![png](output_2_10.png)|
|Right camera|![png](output_2_11.png)|![png](output_2_12.png)|![png](output_2_13.png)|

<hr>

###### Brightness adjustment
* To reduce sensitivity to scene brightness (presumably sunny vs gloomy skies), we add copies of image data with brightness reduced by a factor. Since the data set is bright to begin with, we only reduce brightness, not increase it.

|Source | 75% | 50% | 25% |
|-------|------|------|-------|
|Center camera|![png](output_2_14.png)|![png](output_2_15.png)|![png](output_2_16.png)|
|Left camera|![png](output_2_17.png)|![png](output_2_18.png)|![png](output_2_19.png)|
|Right camera|![png](output_2_20.png)|![png](output_2_21.png)|![png](output_2_22.png)|


<hr>


###### Synthetic shadows
* On track 1, the car encounters shadows of trees, power lines and boulders.
* To train the model to ignore shadows, we add random shadows to training data.

|Source |  Shadow 1    |  Shadow 2 |
|-------|--------------|-----------|
|Center camera|![png](output_2_23.png)|![png](output_2_24.png)|
|Left camera|![png](output_2_25.png)|![png](output_2_26.png)|
|Right camera|![png](output_2_27.png)|![png](output_2_28.png)|


<hr>



## Training
* Multi-process system to augment data in parallel to training on GPU.
* This was necessary to speed up data augmentation and training the model.
* This ensures the 8 cores on the CPU and the GPU were fully utilized.
* Training data is shuffled after each epoch

![png](training.png)

* Training data consists of rows of csv data that is collected by
 - removing 75% of 0-steering data points
 - extracting data for validation
* Training data is shuffled and divided into 8 sets that are fed into a multiprocessing queue.
* Keras was set to spawn and run 8 subprocesses of the data generator.
* Each data generator instance would pop one data set from the multiprocessing queue and use it as its source of images and begin generating augmented images from it.



## Live Trainer
* After training the model, it still requires additional data at places on the track where it veers too close to the edge.
* Drive.py was modified to do live training.

![png](live_training.png)

** Prediction
    * Prediction requests from the simulator are forwarded to a new process 'predict-train.py' which runs on the linux box with the GPU.
    * Throttle values

** Training
    * Drive.py has a UI window that accepts key presses: up,down,left and right
    * Up and down adjust the target speed value (between 0 and 30)
    * If Left/Right key is pressed, drive.py will:
        - calculate new steering value by adusting current steering based on the key pressed
        - send the current image and computed steering to predict-train
        - predict-train will cache the image-steering set and once it has enough cached items, will train the model and update the model file on disk.

### ToDo: Screenshot of live training, showing all UIs

Driving a few laps and training an additional 100 samples corrected the model enough to be able to drive around the first track all night long without crashing.


### ToDo: Video of training lap spedforward


## Attempts


### Training on Grayscale data

### Experimenting with Augmentation
* Steering adjustment for left and right camera images

### Experimenting with Dropout
* No Dropout
* Dropout only on the fully-connected layer
* Dropout on all layers
* Little dropout / Large dropout

### Experimenting with Architectures
* Number of variables same as NVIDIA model (250000)
* border-mode "Valid" and "Same"
* Same: Trainable params: 270,219
* Switching Conv. 3,4 and 5 to use "Valid" mode almost doubled the number of trainable params to 501,819. Models using this mode trained better with lower loss values, but was prone to overfitting, requiring the addition of dropout layers.


### Experimenting with Optimizers


### Experimenting with RELU vs ELU

## Material Referenced




