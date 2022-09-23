import tensorflow as tf
from PIL import Image
import numpy as np
import os

#这些代码相当于 keras里面的load_data函数  只是这个可以用来调用自己的数据集
#
# trainpath = './mnist_image_label/mnist_train_jpg_60000/'  # 训练图片路径
# traintxt = './mnist_image_label/mnist_train_jpg_60000.txt'  # 签路径
# x_train_savepath = './mnist_image_label/mnist_x_train.npy'  # 训练结果保存路径
# y_train_savepath = './mnist_image_label/mnist_y_train.npy'  # 训练结果保存路径（标签）
#
# testpath = './mnist_image_label/mnist_test_jpg_10000/'  # 测试图片路径
# testtxt = '/mnist_image_label/mnist_test_jpg_10000.txt'  # 测试标签路径
# x_test_savepath = './mnist_image_label/mnist_x_test.npy'  # 测试结果保存路径
# y_test_savepath = './mnist_image_label/mnist_y_test.npy'  # 测试结果保存路径（标签）


#
# trainpath = 'E:/桌面/zhenduan/images/training/'  # 训练图片路径
# traintxt = 'E:/桌面/zhenduan/images/training.txt'  # 签路径

trainpath = 'E:/桌面/zhenduan/images/'  # 训练图片路径
traintxt = 'E:/桌面/zhenduan/images/training.txt'  # 签路径
x_train_savepath = 'E:/桌面/zhenduan/images/my_data_x_train.npy'  # 训练结果保存路径
y_train_savepath = 'E:/桌面/zhenduan/images/my_data_y_train.npy'  # 训练结果保存路径（标签）

# testpath = 'E:/桌面/zhenduan/images/testing/'  # 测试图片路径
# testtxt = 'E:/桌面/zhenduan/images/testing.txt'  # 测试标签路径
testpath = 'E:/桌面/zhenduan/images/'  # 测试图片路径
testtxt = 'E:/桌面/zhenduan/images/testing.txt'  # 测试标签路径
x_test_savepath = 'E:/桌面/zhenduan/images/my_data_x_test.npy'  # 测试结果保存路径
y_test_savepath = 'E:/桌面/zhenduan/images/my_data_y_test.npy'  # 测试结果保存路径（标签）

def hanshu(path, txt):  # 定义读取本地数据集函数，其作用是代替之前的load data功能
    file = open(txt, 'r')  # 文件以只读形式打开
    neirong = file.readlines()  # 读出所有行
    file.close()  # 文件关闭

    x, y_ = [], []
    for content in neirong:  # 逐行读出
        values = content.split()  # 将每行的内容以空格分开形成两列 图片路径为value【0】 标签路径为value【1】    文件和图片名字不能有空格
        image_path = path + values[0]  # 读取路径等于图片路径＋图片特征
        img = Image.open(image_path)  # 打开图片

        #img = np.expand_dims(img, axis=2)              #加一个维度
        #img = np.array(img.convert('L'))  # 转化为0-255灰度值的np.array格式    灰度图
        img = np.array(img.convert('RGB'))     #图片是彩色的 设置成RGB可以训练
        img = img / 255  # 归一化
        x.append(img)  # 将值填入x
        y_.append(values[1])  # 将值填入y_
        print('loading : ' + content)  # 打印信息

    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.int64)  # 转换数据类型为64位整形
    return x, y_  # 返回输入特征x，标签y_


if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
        x_test_savepath) and os.path.exists(y_test_savepath):
    print('------load datasets------')
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    # x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))
    # x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))
    x_train = np.reshape(x_train_save, (len(x_train_save), 875, 656))
    x_test = np.reshape(x_test_save, (len(x_test_save), 875, 656))
else:
    print('------making datasets------')
    x_train, y_train = hanshu(trainpath, traintxt)
    x_test, y_test = hanshu(testpath, testtxt)

    print('------saving datasets------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])




model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
model.summary()