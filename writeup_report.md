# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center]: ./center.jpg "Center"
[left]: ./left.jpg "Left"
[right]: ./right.jpg "Right"
[right_turn]: ./right_turn.jpg "Right Turn"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 file showing the car driving a complete lap around track 1

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 26-44) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. A Keras cropping 2D layer is also used to remove irrelevant portions of the input data. The top 50 pixels and bottom 20 pixels of each layer is cropped.

#### 2. Attempts to reduce overfitting in the model

I used a training/validation/test dataset split to ensure I was not overfitting. I ensured this by not allowing the training error to be lower than the validation error. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The error metric used was MSE.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used all three cameras from the sample data set with a steering correction of 0.2 radians applied to the left and right camera images. I also used several custom training runs of my own to teach the network about the tricky curve right after the bridge.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to copy the architecture from the Nvidia paper at https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf, since it was known to work well.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The validation error was consistently less than the training error, so I concluded that I did not need to worry about overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, in particular the left turn after the bridge with the misleading dirt road leading off to the right. To improve the driving behavior in these cases, I generated several training runs showing examples of how to successfully navigate that turn from center-lane driving as well as correctional driving.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
_________________________________________________________________


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center. This image shows the start of a recovery from the right

![alt text][right]

I also captured training data to specifically teach the model about a particularly tricky turn. This is an example of that data:
![alt text][right_turn]

To augment the data sat, I also randomly flipped images and angles to prevent left-turn bias.

I preprocessed the data by cropping to a letterbox and converting the color mode to YUV. I finally randomly shuffled the data set and put 20% of the data into a validation set and 10% into a test set. 
