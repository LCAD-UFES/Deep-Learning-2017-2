# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:33:14 2017
@author: Evandro Dessani
"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
import tensorflow as tf
from keras.utils import multi_gpu_model
import numpy as np
import argparse
from keras.preprocessing.image import ImageDataGenerator

width, height, batch = 32, 32, 8
sgd = SGD(lr=0.01, decay=5e-4, momentum=0.9, nesterov=True)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpus", type=int, default=1,
	help="# of GPUs to use for training")
args = vars(ap.parse_args())
 
# grab the number of GPUs and store it in a conveience variable
G = args["gpus"]

if G <= 1:
	print("[INFO] training with 1 GPU...")
        with tf.device('/device:GPU:1'):
                
            train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
            test_datagen = ImageDataGenerator(rescale = 1./255)
                
            training_set = train_datagen.flow_from_directory('dataset/training_set', target_size = (width, height), batch_size = batch, class_mode = 'categorical')
            test_set = test_datagen.flow_from_directory('dataset/test_set', target_size = (width, height), batch_size = batch, class_mode = 'categorical')
                
            classifier = Sequential()
            classifier.add(Conv2D(filters = 64, kernel_size = (3, 3), padding='same', strides=(1,1), input_shape = (width, height, 3), activation = 'relu'))
            classifier.add(Conv2D(filters = 64, kernel_size = (3, 3), padding='same', strides=(1,1), activation = 'relu'))
            classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2)))
                
            classifier.add(Conv2D(filters = 128, kernel_size = (3, 3), padding='same', strides=(1,1), activation = 'relu'))
            classifier.add(Conv2D(filters = 128, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
            classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2)))
            
            classifier.add(Conv2D(filters = 256, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
            classifier.add(Conv2D(filters = 256, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
            classifier.add(Conv2D(filters = 256, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
            classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2)))
            
            classifier.add(Conv2D(filters = 512, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
            classifier.add(Conv2D(filters = 512, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
            classifier.add(Conv2D(filters = 512, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu')) 
            classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2)))
            
            classifier.add(Conv2D(filters = 512, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
            classifier.add(Conv2D(filters = 512, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
            classifier.add(Conv2D(filters = 512, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
            classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2)))
            
            classifier.add(Flatten())
            
            classifier.add(Dense(units = 4096, activation = 'relu'))
            classifier.add(Dropout(0.5))
            classifier.add(Dense(units = 4096, activation = 'relu'))
            classifier.add(Dropout(0.5))
            classifier.add(Dense(units =3, activation = 'softmax'))
            
            classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
            
            classifier.fit_generator(training_set, steps_per_epoch = 1000, epochs = 5, validation_data = test_set, validation_steps = 250,  use_multiprocessing=True)
    
else: 
    print("[INFO] training with {} GPUs...".format(G))
    with tf.device("/cpu:0"):
        train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
        test_datagen = ImageDataGenerator(rescale = 1./255)
        
        training_set = train_datagen.flow_from_directory('dataset/training_set', target_size = (width, height), batch_size = batch, class_mode = 'categorical')
        test_set = test_datagen.flow_from_directory('dataset/test_set', target_size = (width, height), batch_size = batch, class_mode = 'categorical')
        
        classifier = Sequential()
        classifier.add(Conv2D(filters = 64, kernel_size = (3, 3), padding='same', strides=(1,1), input_shape = (width, height, 3), activation = 'relu'))
        classifier.add(Conv2D(filters = 64, kernel_size = (3, 3), padding='same', strides=(1,1), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2)))
        
        classifier.add(Conv2D(filters = 128, kernel_size = (3, 3), padding='same', strides=(1,1), activation = 'relu'))
        classifier.add(Conv2D(filters = 128, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2)))
        
        classifier.add(Conv2D(filters = 256, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
        classifier.add(Conv2D(filters = 256, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
        classifier.add(Conv2D(filters = 256, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2)))
        
        classifier.add(Conv2D(filters = 512, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
        classifier.add(Conv2D(filters = 512, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
        classifier.add(Conv2D(filters = 512, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu')) 
        classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2)))
        
        classifier.add(Conv2D(filters = 512, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
        classifier.add(Conv2D(filters = 512, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
        classifier.add(Conv2D(filters = 512, kernel_size = (3, 3), padding='same', strides=(1,1), activation= 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2,2)))
        
        classifier.add(Flatten())
        
        classifier.add(Dense(units = 4096, activation = 'relu'))
        classifier.add(Dropout(0.5))
        classifier.add(Dense(units = 4096, activation = 'relu'))
        classifier.add(Dropout(0.5))
        classifier.add(Dense(units =3, activation = 'softmax'))
        
    classifier = multi_gpu_model(classifier, gpus=3)
    
    classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    print("[INFO] compiling model...")
    classifier.fit_generator(training_set, steps_per_epoch = 1000, epochs = 5, validation_data = test_set, validation_steps = 250,  use_multiprocessing=True)

#import numpy as np
#from keras.preprocessing import image
#test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#result = classifier.predict(test_image)
#training_set.class_indices
#if result[0][0] ==  1:                
#    prediction = 'cone'
#else:
#    prediction = 'cat'
#    
## Save model
#classifier.save('cat_or_dogs_modelAcc8941.h5')
#print("Model saved")    
    
