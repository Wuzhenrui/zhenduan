# coding=utf-8
from keras.utils import to_categorical
from keras.datasets import mnist,fashion_mnist,cifar10,imdb
import random
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from  read_my_data import  hanshu

#载入各个数据集
#load.data 会返回 training_data  training_lable  test_data  test_lable
#下面  y是lable   x是data

trainpath = 'E:/桌面/zhenduan/images/'  # 训练图片路径
traintxt = 'E:/桌面/zhenduan/images/training.txt'  # 签路径

testpath = 'E:/桌面/zhenduan/images/'  # 测试图片路径
testtxt = 'E:/桌面/zhenduan/images/testing.txt'  # 测试标签路径

def load_mydata():    #处理我自己的数据集      图片是三维的  卷积输入必须是四维  要加一维

    # (174, 656, 875)(174, )(75, 656, 875)(75, )
    # (173, 656, 875)(173, )
    #(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    (x_train , y_train) = hanshu(trainpath, traintxt)
    (x_test, y_test) = hanshu(testpath, testtxt)
    # x_train = np.expand_dims(x_train, axis=3)              #加一个维度
    #y_train = y_train.reshape(y_train.shape[0], )
    # x_test = np.expand_dims(x_test, axis=3)
    #y_test = y_test.reshape(y_test.shape[0], )
    #
    # x_train = x_train.reshape(-1, 656, 875, 1)
    # y_train = y_train.reshape(y_train.shape[0], )
    # x_test = x_test.reshape(-1, 656, 875, 1)
    # y_test = y_test.reshape(y_test.shape[0], )
    # # x_train = x_train.reshape(-1, 28, 28, 1)
    # # y_train = y_train.reshape(y_train.shape[0], )
    # # x_test = x_test.reshape(-1, 28, 28, 1)
    # # y_test = y_test.reshape(y_test.shape[0], )
    # x_train = x_train / 255.
    # x_test = x_test / 255.
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def load_famnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()    #数据集里面的图就是28*28的
    x_train = x_train.reshape(-1, 28, 28, 1)
    y_train = y_train.reshape(y_train.shape[0], )
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_test = y_test.reshape(y_test.shape[0], )
    x_train = x_train / 255.
    x_test = x_test / 255.
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)        #数据集里面的图就是28*28的
    y_train = y_train.reshape(y_train.shape[0], )
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_test = y_test.reshape(y_test.shape[0], )
    x_train = x_train / 255.
    x_test = x_test / 255.
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

def load_cifar10():          #图片
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(-1, 32, 32, 3)
    y_train = y_train.reshape(y_train.shape[0], )
    x_test = x_test.reshape(-1, 32, 32, 3)
    y_test = y_test.reshape(y_test.shape[0], )
    x_train = x_train / 255.
    x_test = x_test / 255.
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

def load_imdb():    #文本数据集
    config = [5000, 500]
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=config[0])
    x_train = pad_sequences(x_train, maxlen=config[1])
    x_test = pad_sequences(x_test, maxlen=config[1])
    return x_train, y_train, x_test, y_test

def load_data(data_name):
    if data_name == 'famnist':
        x_train, y_train, x_test, y_test = load_famnist()
    elif data_name == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist()
    elif data_name == 'cifar10':
        x_train, y_train, x_test, y_test = load_cifar10()

    elif data_name == 'mydata':
        x_train, y_train, x_test, y_test = load_mydata()
    else:
        x_train, y_train, x_test, y_test = load_imdb()

    return x_train, y_train, x_test, y_test


def get_imb_data(x_train, y_train, x_test, y_test, imb_rate, min_class, maj_class):#--------------------------------读图片 这里分了两类
    # 路径+标签  imb_rat 不平衡率   两个不平衡类


    maj_x_train = []
    maj_y_train = []
    min_x_train = []
    min_y_train = []
    #     print(min_class, maj_class)
    for i in range(len(y_train)):
        if y_train[i] in min_class:
            min_x_train.append(x_train[i])
            min_y_train.append(0)

        if y_train[i] in maj_class:
            maj_x_train.append(x_train[i])
            maj_y_train.append(1)
    #
    min_len = int(len(maj_y_train) * imb_rate)
    new_x_train = maj_x_train + min_x_train[:min_len]             #可能是打乱数据顺序
    new_y_train = maj_y_train + min_y_train[:min_len]
    #     print(len(new_y_train),len(new_y_train))
    #     print(len(maj_x_train))
    new_x_test = []
    new_y_test = []

    for i in range(len(y_test)):
        if y_test[i] in min_class:
            new_x_test.append(x_test[i])
            new_y_test.append(0)
        if y_test[i] in maj_class:
            new_x_test.append(x_test[i])
            new_y_test.append(1)

    new_x_train = np.array(new_x_train)
    new_y_train = np.array(new_y_train)
    new_x_test = np.array(new_x_test)
    new_y_test = np.array(new_y_test)

    idx = [i for i in range(len(new_y_train))]
    np.random.shuffle(idx)

    new_x_train = new_x_train[idx]
    new_y_train = new_y_train[idx]

    return new_x_train, new_y_train, new_x_test, new_y_test
