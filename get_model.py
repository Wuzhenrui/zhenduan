#coding=utf-8
import keras
import tensorflow as tf
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Embedding
from keras.optimizers import Adam, SGD
from keras.layers import LSTM

def get_text_model(input_shape,output):
    top_words, max_words = input_shape
    model = Sequential()
    model.add(Embedding(top_words, 128, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(250))
    model.add(Activation('relu'))
    model.add(Dense(output))
    return model


def get_image_model(in_shape, output):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='Same', input_shape=in_shape))      #这里的32是files是卷积核个数
                                                                            #（5，5）kernel_size 参数 表示卷积核的大小
                                        #padding 是否对周围进行填充，“same” 即使通过kernel_size 缩小了维度，但是四周会填充 0，保持原先的维度
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5), padding='Same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(output))
    return model


