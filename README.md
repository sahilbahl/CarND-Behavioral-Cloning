# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

In this project, I used deep neural networks and convolutional neural networks to clone driving behavior. I trained, validated and tested the model using Keras. The model outputs a steering angle to an autonomous vehicle.

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with similar same as Nvidia's architecture .  

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer . 

#### 2. Attempts to reduce overfitting in the model
To prevent overfitting I first generalized the model by adding flipped images and images from multiple cameras . 

The model was trained and validated with a 0.2 split and trained on 38572 samples, validated on 9644 samples

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually .

#### 4. Appropriate training data

I used training data provided by udacity . I also tried using my own data when the car was unable to stay on the track but later for the final model just used the provided data .  

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a convolution neural network model similar to the Nvidia's paper on end to end deep sdc  using  deep learning . 

However to  be able to improve the model I used all images as well as  flipped images to generalise the model.
 
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low  comparable mean squared error on  training set as well as the validation set. This implied that the model was not overfitting.
I ran for 2 epochs 

Epoch 1/2
38572/38572 [==============================] - 435s - loss: 0.0325 - val_loss: 0.0276

Epoch 2/2
38572/38572 [==============================] - 416s - loss: 0.0252 - val_loss: 0.0265
 


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track [ Near the bridge , After crossing the  bridge and sharp right turn ahead of the bridge ] to improve the driving behavior in these cases, I had to tweak to steering angle as well as add flipped images 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes

Lambda(lambda x: x/255 - 0.5 ,input_shape=(160,320,3) <-Normalization Layer
Cropping2D(cropping=((70,25), (0,0))  <- Cropping Image
Convolution2D(24,5,5,subsample=(2,2),activation="elu")  <- Convolutional Layer
Convolution2D(36,5,5,subsample=(2,2),activation="elu")  <- Convolutional Layer
Convolution2D(48,5,5,subsample=(2,2),activation="elu")  <- Convolutional Layer
Convolution2D(64,3,3,activation="elu")  <- Convolutional Layer
Convolution2D(64,3,3,activation="elu")  <- Convolutional Layer
Flatten()  <- Flattening 
Dense(100) <- Fully-connected layer
Dense(50)  <- Fully-connected layer
Dense(10)  <- Fully-connected layer
Dense(1)   <- Output


#### 3. Creation of the Training Set & Training Process

I used the data provided by Udacity .

To augment the data sat, I also flipped images and angles thinking that this would generalise and prvent the model to be biased to left .

After the collection process, I had 48216 number of data points after using all camera images and fliiping all images and adjusting the left and right angles. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 
