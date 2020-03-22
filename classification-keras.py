# -*- coding: utf-8 -*-
__author__ = 'Aamir Ahmed'
__copyright__ = 'A.A.'

"""
Deep neural network for Classification of hand written digits using Keras and MNIST dataset
"""

# importing libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import mnist
import random
import numpy as np

# read the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# to visualize the image in traing/test dataset 
plt.imshow(random.choice(X_train))
plt.title('random image from training set')
plt.show()
plt.imshow(random.choice(X_test))
plt.title('random image from test set')
plt.show()

X_testunflat = X_test.copy()  # keep original for imshow later

# flatten images into one-dimensional vector
num_pixels = X_train.shape[1] * X_train.shape[2] # find size of one-dimensional vector
X_train = X_train.reshape(X_train.shape[0], num_pixels) # flatten training images
X_test = X_test.reshape(X_test.shape[0], num_pixels) # flatten test images
print(X_test)


# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]
print(num_classes)


# define classification model
def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# build the model
model = classification_model()

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)

print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))

plt.imshow(X_testunflat[249])  # index number for visualization
plt.title("A Digit Image to Recognize as a Test (predicted digit in the shell)")
sample = X_test[:]
print("sample.shape=", sample.shape)
prediction = model.predict(sample)
print("prediction=", np.argmax(prediction[249]))  # index number for prediction
plt.show()
