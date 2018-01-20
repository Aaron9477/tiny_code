import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

import numpy as np

model = Sequential()
model.add(Dense(32,activation='relu', input_dim=100))
model.add(Dense(1,activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

data = np.random.random((1000,100))
lables = np.random.randint(2,size=(1000,1))

model.fit(data, lables, epochs=10, batch_size=32)