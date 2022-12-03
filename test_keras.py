import os
import sys
import numpy as np
import tensorflow as tf
from numpy import ndarray
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
    Concatenate,  # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Concatenate
)

from freqtrade.configuration import TimeRange
from freqtrade.data.history import load_data
from freqtrade.enums import CandleType
from pathlib import Path

pair_list = [
    'BTC/USDT',
]  # XRP/USDT ETH/USDT TRX/USDT

data = load_data(
    datadir=Path('./freqtrade/user_data/data/binance'),
    pairs=pair_list,
    timeframe='5m',
    timerange=TimeRange.parse_timerange('20210101-20220101'),
    startup_candles=0,
    data_format='jsongz',
    candle_type=CandleType.FUTURES,
)


dataframe = data['BTC/USDT']
dataframe = dataframe.drop(['date'], axis=1)
mask_volume = (dataframe['volume'] > 0)

window_backward = 200
import generate_answer
# answer = generate_answer.generate_answer(dataframe['close'].to_numpy(), window_backward=1,
                                                      # window_forward=200)
answer = generate_answer.generate_answer_v2(dataframe['close'].to_numpy(), threshold=0.02, enum_unknown=-1,
                                            enum_up=1, enum_down=0)
mask_answer = (answer != -1)

import indicator
dataframe['heikin-ashi_close'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
dataframe['moving_average_simple_200'] = indicator.moving_average_simple(dataframe['heikin-ashi_close'].to_numpy(), window=200)
dataframe['regression_1_200'] = indicator.regression_1(dataframe['heikin-ashi_close'].to_numpy(), window=200)

import talib.abstract as ta
dataframe['EMA200'] = ta.EMA(dataframe['heikin-ashi_close'], timeperiod=200)
dataframe['WMA200'] = ta.WMA(dataframe['heikin-ashi_close'], timeperiod=200)
# dataframe'RSI9'] = ta.RSI(dataframe['heikin-ashi_close'], timeperiod=9)

import freqtrade.vendor.qtpylib.indicators as qtpylib
dataframe['HMA200'] = qtpylib.hma(dataframe['heikin-ashi_close'], window=200)

column_drop = ['open', 'high', 'low', 'close', 'volume']
column_drop.append('heikin-ashi_close')
dataframe = dataframe.drop(column_drop, axis=1)

# column_drop = ['volume']
# dataframe = dataframe.drop(column_drop, axis=1)

column = dataframe.columns.tolist()
print(column)
# sys.exit()

# from sklearn import preprocessing
# # for i in ['open', 'high', 'low', 'close', 'volume']:
# for i in column:
    # dataframe[i] = preprocessing.scale(dataframe[i] - dataframe[i].shift(1))

for i in range(len(column)):
    for j in range(i + 1, len(column)):
        dataframe[f'{column[i]} - {column[j]}'] = dataframe[column[i]] - dataframe[column[j]]
dataframe = dataframe.drop(column, axis=1)
print(f'column: {dataframe.columns.tolist()}')


# print(dataframe)
# sys.exit()

import generate_window
target, mask_final = (
    generate_window.generate_window_v2(
        dataframe.to_numpy(), (mask_volume & mask_answer).to_numpy(), window=window_backward, exclude_nan=True
    )
)

import reduce_mask
answer = reduce_mask.reduce_mask(answer, mask_final)

if len(target) != len(answer):
    raise Exception

# print(target[:20], answer[:20])
# sys.exit()

# ratio = 0.7
# border = int(len(target) * ratio)
length_margin = 1600
length_test = 400
length_train = 80000
start_train = len(target) - (length_test + length_train) - length_margin
print(f'ratio: {length_train / length_test:0.4f}')

from tensorflow.keras.utils import to_categorical
answer = to_categorical(answer)
# print(answer)

# target_train = target[:border]
# target_test  = target[border:]
# answer_train = answer[:border]
# answer_test  = answer[border:]

target_train = target[-(length_train + length_test):-length_test]
target_test  = target[-length_test:]
answer_train = answer[-(length_train + length_test):-length_test]
answer_test  = answer[-length_test:]

print(f'length: all: {len(target)}')
print(f'length: train: {len(target_train)}')
print(f'length: test: {len(target_test)}')

x_train = target_train
x_test  = target_test
y_train = answer_train
y_test  = answer_test

data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
data_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# The batch size must now be set on the Dataset objects.
batch_size = 200
data_train = data_train.batch(batch_size)
data_test = data_test.batch(batch_size)

# Disable AutoShard.
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
data_train = data_train.with_options(options)
data_test = data_test.with_options(options)

def print_summary_y(name: str, a: ndarray):
    for i in [0, 1, 2]:
        print(f'{name}: {i}: {np.count_nonzero(a == i) / len(a) * 100:0.4f}(%)')

print_summary_y('All of answer', answer)
print_summary_y('Train part of answer', answer_train)
print_summary_y('Test part of answer', answer_test)

# sys.exit()

from keras_layer import RelativePosition, relative_position, DenseInputBias

class DenseBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(DenseBlock, self).__init__()
        self.layer_1 = Flatten()
        self.layer_2 = Dense(64)
        self.layer_3 = BatchNormalization()
        self.layer_4 = Activation('relu')
        self.layer_5 = Dense(64)
        self.layer_6 = Activation('relu')
        self.layer_7 = BatchNormalization()
        self.layer_8 = Activation('relu')
        self.layer_9 = Dense(4)

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

    # https://www.tensorflow.org/api_docs/python/tf/keras/Model
    inputs = Input(shape=(200, 10,))

    b1 = DenseBlock()(inputs)
    b2 = DenseBlock()(inputs)
    b3 = DenseBlock()(inputs)
    b4 = DenseBlock()(inputs)
    b5 = DenseBlock()(inputs)
    b6 = DenseBlock()(inputs)
    b7 = DenseBlock()(inputs)
    b8 = DenseBlock()(inputs)
    b9 = DenseBlock()(inputs)
    b10 = DenseBlock()(inputs)
    b11 = DenseBlock()(inputs)
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
    x = Dense(5000)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(5000)(x)
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

if 'POPLAR_SDK_ENABLED' in os.environ:
    from tensorflow.python import ipu
    ipu_config = ipu.config.IPUConfig()
    ipu_config.auto_select_ipus = 16
    ipu_config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()
else:
    gpu = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpu)

print(f'device: {tf.config.list_logical_devices()}')

if False:
    tf.debugging.set_log_device_placement(True)

from tensorflow.keras.optimizers import Adam

with strategy.scope():
    try:
        model = tf.keras.models.load_model('./model')
    except OSError as error:
        print(f'Model has not found: {error}')
        model = define_model()

    early_stopping = EarlyStopping(monitor='loss', patience=100)
    model.compile(optimizer=Adam(learning_rate=1e-6), loss='mse', metrics=['accuracy'])  # sparse_categorical_crossentropy sgd categorical_crossentropy
    model.summary()

    while True:
        try:
            # history = model.fit(x_train, y_train, batch_size=200, epochs=1, validation_data=(x_test, y_test),
            history = model.fit(x_train, y_train, batch_size=200, epochs=1, validation_data=data_test,
                                callbacks=[early_stopping])

            # y_predict_probability = model.predict(x_test)
            # y_predict = np.argmax(y_predict_probability, axis=1)
            # # print(len(y_test), len(y_predict))
            # print(f'accuracy: {(np.count_nonzero(y_test == y_predict)) / len(y_predict) * 100:0.4f}(%)')
            # for i in [0, 1, 2]:
                # print(f'accuracy: {i}: {np.count_nonzero((y_test == y_predict) & (y_predict == i)) / len(y_predict) * 100:0.4f}(%)')

        except KeyboardInterrupt:
            print(f'\nPaused: KeyboardInterrupt')
            model.save('./model')
            break
