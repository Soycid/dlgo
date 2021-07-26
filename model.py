from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from sklearn.model_selection import train_test_split
import sgf2vec
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(np.array(sgf2vec.X), np.array(sgf2vec.y), test_size=.33)

#y_train = keras.utils.to_categorical(y_train, 19*19)
#y_test = keras.utils.to_categorical(y_test, 19*19)

model = Sequential()


def layers(input_shape):
    return [
        ZeroPadding2D((3, 3), input_shape=input_shape, data_format='channels_first'),
        Conv2D(64, (7, 7), padding='valid', data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format='channels_first'),
        Conv2D(64, (5, 5), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format='channels_first'),
        Conv2D(64, (5, 5), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format='channels_first'),
        Conv2D(48, (5, 5), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format='channels_first'),
        Conv2D(48, (5, 5), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format='channels_first'),
        Conv2D(32, (5, 5), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format='channels_first'),
        Conv2D(32, (5, 5), data_format='channels_first'),
        Activation('relu'),

        Flatten(),
        Dense(19*19,activation='softmax'),
        Reshape((19,19))

    ]

#import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

print("yeet")
network_layers = layers((6,19,19))
for layer in network_layers:
    model.add(layer)
#num_classes = 19*19
#model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy'])

model.fit(x_train, y_train,
        batch_size=8,
        epochs=20)

score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])