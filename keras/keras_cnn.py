from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt

# batch_size : 每个迭代的样本数量
batch_size = 128
# num_classes : 输出样本个数
num_classes = 10
# epochs : 所有样本都跑一遍
epochs = 10

# input image dimensions
img_x, img_y = 28, 28

# load the MNIST data set, which already splits into train and test sets for us
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

# convert the data to the right type
# .astype(type) -> 强制类型转换
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
# 2D卷积层: 输出通道32 核5x5 步态1x1 激活函数relu
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))

# 2D MaxPooling层：池大小2x2 步态2x2
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 卷积层+最大池化层
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 输出平坦化
model.add(Flatten())

# 全连接层 ：1000个节点
model.add(Dense(1000, activation='relu'))

# softmax分类输出层
model.add(Dense(num_classes, activation='softmax'))

# 损失函数 : 标准交叉熵
# 优化器 : adam
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()


# 训练
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          # verbose = 1 : 指定设定是否要在控制台中打印详细信息以了解训练进度
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])


# 将验证或测试数据传递给拟合函数
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
