import os
import sys
import numpy as np
import tensorflow as tf
from numpy import ndarray
from pandas import DataFrame
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling1D,
    UpSampling1D,
    GRU,
    Input,
    Concatenate,
)


import tensorflow
import tensorflow.experimental.numpy as tnp

tnp.experimental_enable_numpy_behavior()
jit = tensorflow.function(jit_compile=True)

@jit
def loss_custom(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor) -> tensorflow.Tensor:
    batch_size = y_pred.shape[0]
    y_true = tensorflow.cast(y_true, y_pred.dtype)

    if y_pred.shape != (batch_size, 3):
        raise Exception(y_pred.shape)
    if y_true.shape != (batch_size, 2):
        raise Exception(y_true.shape)

    y_pred_index_maximum = tnp.argmax(y_pred, axis=1)
    y_true_index_maximum = tnp.argmax(y_true, axis=1)

    y_pred = y_pred[:, :2]
    mask = tnp.repeat((y_pred_index_maximum == y_true_index_maximum), 2).reshape(batch_size, 2)

    point_plus = y_pred[mask & (y_true == 1)]
    point_minus = -y_pred[mask & (y_true == 0)]

    loss = batch_size - (tnp.sum(point_plus) + tnp.sum(point_minus))
    return loss

print(loss_custom(tnp.array([[1, 0], [1, 0], [1, 0]]), tnp.array([[0.1, 0.3, 0.6], [0.1, 0.6, 0.3], [0.6, 0.3, 0.1]])))
assert loss_custom(tnp.array([[1, 0], [1, 0], [1, 0]]), tnp.array([[0.1, 0.3, 0.6], [0.1, 0.6, 0.3], [0.6, 0.3, 0.1]])) == 2.7

def reverse_direction(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor) -> tensorflow.Tensor:
    batch_size = y_pred.shape[0]
    y_true = tensorflow.cast(y_true, y_pred.dtype)

    if y_pred.shape != (batch_size, 3):
        raise Exception(y_pred.shape)
    if y_true.shape != (batch_size, 2):
        raise Exception(y_true.shape)

    y_pred_index_maximum = tnp.argmax(y_pred, axis=1)
    y_true_index_maximum = tnp.argmax(y_true, axis=1)

    # mask0 = (y_true_index_maximum == y_pred_index_maximum) & (y_true_index_maximum == 0)
    # ratio0 = tnp.count_nonzero(mask0) / len(y_true_index_maximum)
    mask0 = (y_true_index_maximum != y_pred_index_maximum) & (y_pred_index_maximum != 2)
    ratio0 = tnp.count_nonzero(mask0) / len(y_true_index_maximum)
    return ratio0

print(reverse_direction(tnp.array([[1, 0], [1, 0], [1, 0]]), tnp.array([[0.1, 0.3, 0.6], [0.1, 0.6, 0.3], [0.6, 0.3, 0.1]])))
# sys.exit()


from load_data import load_data

x_train, y_train, x_test, y_test = load_data()
print(x_train.shape, x_test.shape)
print(x_train)

y_train = tensorflow.keras.utils.to_categorical(y_train)
y_test = tensorflow.keras.utils.to_categorical(y_test)

print(y_train.shape, y_test.shape)
print(y_train)

window = x_train.shape[1]
feature = x_train.shape[2]
# output_class = y_train.shape[1]
output_class = y_train.shape[1] + 1

from tensorflow.keras.layers import Layer
from keras_layer import RelativePosition, relative_position, SimpleDense, DenseAverage, DenseBatchNormalization# , DenseInputBias, DenseAverage

class DenseBlock(Layer):
    def __init__(self):
        super(DenseBlock, self).__init__()
        self.layer_1 = Dense(64)
        self.layer_2 = BatchNormalization()
        self.layer_3 = Activation('relu')
        self.layer_4 = Dense(64)
        self.layer_5 = BatchNormalization()
        self.layer_6 = Activation('relu')
        self.layer_7 = Dense(4)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        return x

class DenseBlockSkip(Layer):
    def __init__(self, n):
        super(DenseBlockSkip, self).__init__()
        self.layer_1 = BatchNormalization()
        self.layer_2 = Activation('relu')
        self.layer_3 = Dense(64)
        self.layer_4 = BatchNormalization()
        self.layer_5 = Activation('relu')
        self.layer_6 = Dense(n)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x += inputs
        return x

def define_model():
    '''
    class CustomModel(tf.keras.Model):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.layer_1 = Conv1D(16, kernel_size=200)
            self.layer_2 = Flatten()
            self.layer_3 = RelativePosition(n=16)
            self.layer_4 = Flatten()
            self.layer_5 = Dense(64)
            self.layer_6 = Activation('relu')
            self.layer_7 = BatchNormalization()
            self.layer_8 = Dense(2)
            self.layer_9 = Activation('softmax')

        def call(self, inputs):
            x = self.layer_1(inputs)
            x = self.layer_2(x)
            x = self.layer_3(x)
            x = self.layer_4(x)
            x = self.layer_5(x)
            x = self.layer_6(x)
            x = self.layer_7(x)
            x = self.layer_8(x)
            x = self.layer_9(x)
            return x

    # inputs = Input(shape=(200, 1,))
    model = CustomModel()  # inputs=inputs, outputs=x
    '''

    inputs = Input(shape=(window, feature))

    '''
    x = Flatten()(inputs)
    b1 = DenseBlock()(x)
    b2 = DenseBlock()(x)
    b3 = DenseBlock()(x)
    b4 = DenseBlock()(x)
    b5 = DenseBlock()(x)
    b6 = DenseBlock()(x)
    b7 = DenseBlock()(x)
    b8 = DenseBlock()(x)
    b9 = DenseBlock()(x)
    b10 = DenseBlock()(x)
    b11 = DenseBlock()(x)
    x = Concatenate()([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11])
    # x = RelativePosition()(x)
    x = BatchNormalization()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(2)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    '''

    x = Flatten()(inputs)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(output_class)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=x)

    '''
    x = Flatten()(inputs)
    x = SimpleDense(1250)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SimpleDense(1250)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SimpleDense(2)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    '''

    '''
    x = Flatten()(inputs)
    x = DenseAverage(1250)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DenseAverage(1250)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SimpleDense(2)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    '''

    '''
    x = Flatten()(inputs)
    x = DenseBatchNormalization(2048)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DenseBatchNormalization(64)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SimpleDense(2)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    '''

    '''
    x = Flatten()(inputs)
    x = Dense(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(2)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    '''

    '''
    model = tf.keras.models.Sequential([
        GRU(64, return_sequences=True),
        Activation('relu'),
        GRU(64, return_sequences=False),
        Activation('relu'),
        Dense(16),
        # BatchNormalization(),
        Activation('relu'),
        Dense(2),
        Activation('softmax'),
    ])
    '''

    return model

from keras_device import scope
from tensorflow.keras.optimizers import Adam

with scope():
    # data_train = strategy.experimental_distribute_dataset(data_train)
    # data_test = strategy.experimental_distribute_dataset(data_test)

    try:
        model = tf.keras.models.load_model('./model')
    except OSError as error:
        print(f'Model has not found: {error}')
        model = define_model()

    early_stopping = EarlyStopping(monitor='loss', patience=10)
    model.compile(optimizer=Adam(), loss=loss_custom, metrics=['accuracy', reverse_direction])  # learning_rate=1e-6 mean_squared_error
    model.summary()

    while True:
        try:
            history = model.fit(x_train, y_train, batch_size=200, epochs=300, validation_data=(x_test, y_test),
            # history = model.fit(data_train, batch_size=batch_size, epochs=1, validation_data=data_test,
                                callbacks=[early_stopping])

        except KeyboardInterrupt:
            print(f'\nPaused: KeyboardInterrupt')
            # model.save('./model')
            break
