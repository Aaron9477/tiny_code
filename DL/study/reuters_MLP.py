#!/usr/bin/env python

import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from keras.datasets import imdb
from keras.datasets import cifar10
from keras.datasets import fashion_mnist

import numpy as np

# data = np.random.random((1000,100))
# test = np.random.randint(2, size=(1000,1))
# print(data)
# exit()

# (x_train, y_train), (x_test, y_test) = reuters.load_data()
# print('imdb: ',x_train.shape, y_train.shape, x_test.shape, y_test.shape)

(x_train, y_train), (x_test, y_test) = imdb.load_data()
print('imdb: ',x_train.shape, y_train.shape, x_test.shape, y_test.shape)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('cifar10: ',x_train.shape, y_train.shape, x_test.shape, y_test.shape)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print('fashion_mnist: ', x_train.shape, y_train.shape, x_test.shape, y_test.shape)








