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
    # 'BTC/USDT',
    'ETH/USDT',
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


dataframe = data['ETH/USDT']
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
length_test = 3200
length_train = 100000
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

# data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# data_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

batch_size = 200
# data_train = data_train.batch(batch_size, drop_remainder=True)
# data_test = data_test.batch(batch_size, drop_remainder=True)

# option = tf.data.Options()
# option.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
# data_train = data_train.with_options(option)
# data_test = data_test.with_options(option)

def make_divisible(number, divisor):
    return number - number % divisor

train_data_len = x_train.shape[0]
train_data_len = make_divisible(train_data_len, batch_size)
x_train, y_train = x_train[:train_data_len], y_train[:train_data_len]

test_data_len = x_test.shape[0]
test_data_len = make_divisible(test_data_len, batch_size)
x_test, y_test = x_test[:test_data_len], y_test[:test_data_len]

print(x_test.shape)
print(x_train.shape)
print(y_test.shape)
print(y_train.shape)


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

class DenseBlockSkip(tf.keras.layers.Layer):
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

    inputs = Input(shape=(200, 10))

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

    '''
    x = Flatten()(inputs)
    x = Dense(1500)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(1500)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(2)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=x)
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
    ipu_config.auto_select_ipus = 1
    ipu_config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()
    strategy_scope = strategy.scope()
else:
    # gpu = tf.config.list_logical_devices('GPU')
    # strategy = tf.distribute.MirroredStrategy(gpu)
    from contextlib import nullcontext
    strategy_scope = nullcontext()

print(f'device: {tf.config.list_logical_devices()}')

if False:
    tf.debugging.set_log_device_placement(True)

from tensorflow.keras.optimizers import Adam


with strategy_scope:
    # data_train = strategy.experimental_distribute_dataset(data_train)
    # data_test = strategy.experimental_distribute_dataset(data_test)

    try:
        model = tf.keras.models.load_model('./model')
    except OSError as error:
        print(f'Model has not found: {error}')
        model = define_model()

    early_stopping = EarlyStopping(monitor='loss', patience=10)
    model.compile(optimizer=Adam(learning_rate=1e-6), loss='mse', metrics=['accuracy'])  # sparse_categorical_crossentropy sgd categorical_crossentropy
    model.summary()

    while True:
        try:
            history = model.fit(x_train, y_train, batch_size=200, epochs=300, validation_data=(x_test, y_test),
            # history = model.fit(data_train, batch_size=batch_size, epochs=1, validation_data=data_test,
                                callbacks=[early_stopping])
                                # callbacks=[early_stopping], steps_per_epoch=x_train.shape[0] // batch_size, validation_steps=x_test.shape[0] // batch_size)

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
