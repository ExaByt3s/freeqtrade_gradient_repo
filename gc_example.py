import os
import time
import numpy as np
import tensorflow
import tensorflow as tf
import tensorflow.keras as keras

num_classes = 10
input_shape = (28, 28, 1)
batch_size = 64

def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

def model_fn():
    input_layer = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)

    return input_layer, x

def make_divisible(number, divisor):
    return number - number % divisor


(x_train, y_train), (x_test, y_test) = load_data()

train_data_len = x_train.shape[0]
train_data_len = make_divisible(train_data_len, batch_size)
x_train, y_train = x_train[:train_data_len], y_train[:train_data_len]

test_data_len = x_test.shape[0]
test_data_len = make_divisible(test_data_len, batch_size)
x_test, y_test = x_test[:test_data_len], y_test[:test_data_len]

scope_all = []
scope_all.append(tensorflow.device('CPU'))

if 'POPLAR_SDK_ENABLED' in os.environ:
    from tensorflow.python import ipu
    ipu_config = ipu.config.IPUConfig()
    ipu_config.auto_select_ipus = 1
    ipu_config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()
    scope_all.append(strategy.scope())

for scope in scope_all:
    print(scope)
    time_begin = time.perf_counter()

    with scope:
        model = keras.Model(*model_fn())
        model.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        model.fit(x_train, y_train, epochs=3, batch_size=batch_size, validation_data=(x_test, y_test))

    time_end = time.perf_counter()
    print(f'{time_end - time_begin:0.4f} (second)')
