# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
# Part 1 - Building the CNN

# Importing the keras libraries and packages

import tensorflow as tf
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import RMSprop
#from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
import json

# Optimizers
sgd = SGD(lr=0.01, decay=1e-2, momentum=0.9, nesterov=True)
rms = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)

   
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Convolution2D(96, 7, 7, border_mode='same', input_shape=(32, 32, 3), activation = 'relu'))
# Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', data_format=None))
# Step 3 - Contrast Image
classifier.add(BatchNormalization())
classifier.compile(optimizer = 'RMSprop', loss = 'categorical_crossentropy')

# Step 4 - Convolution
classifier.add(Convolution2D(256, 5, 5, border_mode='same', activation = 'relu'))
# Step 5 - Max Pooling
classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', data_format=None))
# Step 6 - Contrast Image
classifier.add(BatchNormalization())
classifier.compile(optimizer = 'RMSprop', loss = 'categorical_crossentropy')

# Step 7 - Convolution
classifier.add(Convolution2D(512, 3, 3, border_mode='same', activation = 'relu'))

# Step 8 - Convolution
classifier.add(Convolution2D(1024, 3, 3, border_mode='same', activation = 'relu'))

# Step 9 - Convolution
classifier.add(Convolution2D(512, 3, 3, border_mode='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', data_format=None))

# Step 10 - Flattening
classifier.add(Flatten())

# Step 11  - Full Connection
classifier.add(Dense(output_dim = 4096, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 4096, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 3, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.summary()

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(32, 32),
                                                batch_size=12,
                                                class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(32, 32),
                                            batch_size=12,
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                        steps_per_epoch=12000,
                        epochs=30,
                        validation_data=test_set,
                        validation_steps=3000)
#
# Part 3 - Save the weights
classifier.save('TesteFinal.h5')
print('Classifier Saved')

# Part 4 - Making new pewdictions
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_or_car_3.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(result)
training_set.class_indices

test_image = image.load_img('dataset/single_prediction/cat_or_dog_or_car_2.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(result)
training_set.class_indices

test_image = image.load_img('dataset/single_prediction/cat_or_dog_or_car_1.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(result)
training_set.class_indices

test_image = image.load_img('dataset/single_prediction/cat_or_dog_or_car_4.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(result)
training_set.class_indices

# Part 5 - Save the structure model
classifier_json = classifier.to_json()
with open("AlexNet_ZFnet_CNNStructure.json", "w") as json_file:
    json_file.write(classifier_json)
























