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
    # datadir=Path('/user/one/home/user_data/data/binance'),
    datadir=Path('./freqtrade/user_data/data/binance'),
    # datadir=Path('freqtrade/user_data/data/binance'),
    # datadir=Path('/user/one/home/trade/gradient/freqtrade/user_data/data/binance'),
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
# dataframe['moving_average_simple_100'] = indicator.moving_average_simple(dataframe['heikin-ashi_close'].to_numpy(), window=100)
# dataframe['moving_average_simple_200'] = indicator.moving_average_simple(dataframe['heikin-ashi_close'].to_numpy(), window=200)
# dataframe['regression_1_100'] = indicator.regression_1(dataframe['heikin-ashi_close'].to_numpy(), window=100)
# dataframe['regression_1_200'] = indicator.regression_1(dataframe['heikin-ashi_close'].to_numpy(), window=200)

import talib.abstract as ta
# dataframe['EMA100'] = ta.EMA(dataframe['heikin-ashi_close'], timeperiod=100)
# dataframe['EMA200'] = ta.EMA(dataframe['heikin-ashi_close'], timeperiod=200)
# dataframe['WMA100'] = ta.WMA(dataframe['heikin-ashi_close'], timeperiod=100)
# dataframe['WMA200'] = ta.WMA(dataframe['heikin-ashi_close'], timeperiod=200)
# dataframe'RSI9'] = ta.RSI(dataframe['heikin-ashi_close'], timeperiod=9)

import freqtrade.vendor.qtpylib.indicators as qtpylib
# dataframe['HMA100'] = qtpylib.hma(dataframe['heikin-ashi_close'], window=100)
# dataframe['HMA200'] = qtpylib.hma(dataframe['heikin-ashi_close'], window=200)

column_drop = ['open', 'high', 'low', 'close', 'volume']
# column_drop.append('heikin-ashi_close')
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
'''
for i in range(len(column)):
    for j in range(i + 1, len(column)):
        dataframe[f'{column[i]} - {column[j]}'] = dataframe[column[i]] - dataframe[column[j]]
dataframe = dataframe.drop(column, axis=1)
print(f'column: {dataframe.columns.tolist()}')
'''

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
length_margin = 1800
length_test = 200
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

def print_summary_y(name: str, a: ndarray):
    for i in [0, 1, 2]:
        print(f'{name}: {i}: {np.count_nonzero(a == i) / len(a) * 100:0.4f}(%)')

print_summary_y('All of answer', answer)
print_summary_y('Train part of answer', answer_train)
print_summary_y('Test part of answer', answer_test)

# sys.exit()

from keras_layer import RelativePosition, relative_position, DenseInputBias
def define_model():
    '''
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(2000, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(50, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(3, activation=tf.nn.softmax),
    ])
    '''

    '''
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(200, 1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    '''
    '''
    look_back = 200
    conv_filter = 4
    units = 64
    # batch_size = 8
    # opt = Adam(learning_rate=0.001)
    model = Sequential()
    model.add(Conv1D(filters=conv_filter, kernel_size=1, padding='same', activation='tanh',batch_input_shape=(None, look_back, 1)))
    model.add(MaxPool1D(pool_size=1, padding='same'))
    model.add(Activation('relu'))
    model.add(LSTM(units, return_sequences=True))
    # model.add(Dense(1, kernel_initializer='random_uniform'))
    model.add(Dense(2, kernel_initializer='random_uniform'))
    # model.compile(loss = "mean_absolute_error", optimizer=opt)
    # print(model.summary())
    '''
    '''
    inputs = Input(shape=(784,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(input=inputs, output=predictions)
    '''
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
    inputs = Input(shape=(200, 1,))

    b1 = Flatten()(inputs)
    b1 = Dense(16)(b1)
    # b1 = BatchNormalization()(b1)
    b1 = Activation('relu')(b1)
    b1 = Dense(8)(b1)
    # b1 = BatchNormalization()(b1)
    b1 = Activation('relu')(b1)
    b1 = Dense(1)(b1)

    b2 = Flatten()(inputs)
    b2 = Dense(16)(b2)
    # b2 = BatchNormalization()(b2)
    b2 = Activation('relu')(b2)
    b2 = Dense(8)(b2)
    # b2 = BatchNormalization()(b2)
    b2 = Activation('relu')(b2)
    b2 = Dense(1)(b2)

    b3 = Flatten()(inputs)
    b3 = Dense(16)(b3)
    # b3 = BatchNormalization()(b3)
    b3 = Activation('relu')(b3)
    b3 = Dense(8)(b3)
    # b3 = BatchNormalization()(b3)
    b3 = Activation('relu')(b3)
    b3 = Dense(1)(b3)

    x = Concatenate()([b1, b2, b3])
    x = Flatten()(x)
    x = RelativePosition()(x)
    x = BatchNormalization()(x)
    x = Dense(32)(x)
    # x = DenseInputBias()(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(16)(x)
    # x = DenseInputBias()(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(2)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=x)

    '''
    model = tf.keras.models.Sequential([
        Flatten(),
        Dense(16),
        BatchNormalization(),
        Activation('relu'),
        Dense(2),
        Activation('softmax'),
    ])
    '''

    '''
    model = tf.keras.models.Sequential([
        Flatten(),
        # Dense(512),
        Dense(16),
        BatchNormalization(),
        Activation('relu'),
        # Dropout(0.1),
        # Dense(256),
        # Dense(16),
        # BatchNormalization(),
        # Activation('relu'),
        # Dropout(0.1),
        Dense(2),
        Activation('softmax'),
    ])
    '''

    '''
    model = tf.keras.models.Sequential([
        Flatten(),
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        # Dropout(0.1),
        Dense(64),
        BatchNormalization(),
        Activation('relu'),
        # Dropout(0.1),
        Dense(2),
        Activation('softmax'),
    ])
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

try:
    model = tf.keras.models.load_model('./model')
except OSError as error:
    print(f'Model has not found: {error}')

    model = define_model()

# sys.exit()
from tensorflow.keras.optimizers import Adam
early_stopping = EarlyStopping(monitor='loss', patience=100)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['accuracy'])  # sparse_categorical_crossentropy sgd categorical_crossentropy
model.summary()

while True:
    try:
        history = model.fit(x_train, y_train, batch_size=256, epochs=1, validation_data=(x_test, y_test),
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
