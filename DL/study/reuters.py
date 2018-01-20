#!/usr/bin/env python


import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import RMSprop, Adam


def main():
    (x_train, y_train), (x_test, y_test) = reuters.load_data(path='/home/zq610/WYZ/download/DL_datasets/reuters.npz',
                                                             nb_words=None,
                                                             skip_top=0,
                                                             maxlen=None,
                                                             test_split=0.2,
                                                             seed=1,
                                                             start_char=1,
                                                             oov_char=2,
                                                             index_from=3)
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # print(x_train)
    # print(y_train)
    a = reducelambda
    print(len(x_train[1]))
    # model = Sequential()
    # model.add(Dense(64, activation='relu', input_dim=20))
    # model.


if __name__ == '__main__':
    main()