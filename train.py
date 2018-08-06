import csv
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Dropout
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
import sklearn
from random import shuffle
from sklearn.model_selection import train_test_split
import random
lines = []

with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
images = []
measurements =[]
left_correction = 0.20
right_correction = 0.30
for line in lines:
        centerMeasurement = float(line[3])		
        centerFilePath = line[0].strip()
        leftFilePath = line[1].strip()
        rightFilePath = line[2].strip()
        
        centerImage = plt.imread(centerFilePath)
        leftImage = plt.imread(leftFilePath)
        rightImage = plt.imread(rightFilePath)
        
        #Use center ,left and right images 
        images.append(centerImage)
        images.append(leftImage)
        images.append(rightImage)

        #Flip Imgages to genaralize the model
        images.append(np.fliplr(centerImage))
        images.append(np.fliplr(leftImage))
        images.append(np.fliplr(rightImage))
 
 		#Correction for left and right angle
        leftMeasurement = centerMeasurement + left_correction
        rightMeasurement = centerMeasurement - right_correction
        
        measurements.append(centerMeasurement)
        measurements.append(leftMeasurement)
        measurements.append(rightMeasurement)
        measurements.append(centerMeasurement*-1.0)
        measurements.append(leftMeasurement*-1.0)
        measurements.append(rightMeasurement*-1.0)
        
    
X_train = np.array(images)
Y_train = np.array(measurements)


model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5 ,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="elu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="elu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="elu"))
model.add(Convolution2D(64,3,3,activation="elu"))
model.add(Convolution2D(64,3,3,activation="elu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse',optimizer='adam')
model.fit(X_train,Y_train,validation_split=0.2,shuffle=True,nb_epoch=2)
model.save('model.h5')

exit()
