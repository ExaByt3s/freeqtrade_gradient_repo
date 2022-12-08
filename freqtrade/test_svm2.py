import json
import sys
import numpy as np
# import tensorflow as tf
from numpy import ndarray
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Activation, BatchNormalization, Conv1D, Dense, Dropout, Flatten, MaxPooling1D, UpSampling1D

from freqtrade.configuration import TimeRange
from freqtrade.data.history import load_data
from freqtrade.enums import CandleType
from pathlib import Path

data = load_data(
    datadir=Path('/user/one/home/user_data/data/binance'),
    pairs=['BTC/USDT'],
    timeframe='5m',
    timerange=TimeRange.parse_timerange('20220101-20220802'),
    startup_candles=0,
    data_format='json',
    candle_type=CandleType.FUTURES,
)

dataframe = data['BTC/USDT']

window = 800
import generate_answer
# dataframe['answer'] = generate_answer.generate_answer(dataframe['close'].to_numpy(), window_backward=1,
                                                      # window_forward=20000)
# mask_answer = (dataframe['answer'] != -1)

dataframe['answer'] = generate_answer.generate_answer_v2(dataframe['close'].to_numpy(), threshold=0.01, enum_unknown=-1,
                                                         enum_up=1, enum_down=0)
mask_answer = (dataframe['answer'] != -1)
# print(dataframe['answer'][-100:])

# sys.exit()

# from sklearn import preprocessing
# dataframe['close_normalized'] = preprocessing.scale(dataframe['close'] - dataframe['close'].shift(1))
dataframe['close_normalized'] = dataframe['close'] - dataframe['close'].shift(1)
# print(dataframe['close_normalized'])

import generate_window
target, mask_final = (
    generate_window.generate_window(
        dataframe['close_normalized'].to_numpy(),
        ((dataframe['volume'] > 0) & ~(dataframe['close_normalized'].isna()) & mask_answer).to_numpy(),
        window=window,
    )
)

import reduce_mask
answer = reduce_mask.reduce_mask(dataframe['answer'].to_numpy(), mask_final)

target = target[-4000:-2000]
answer = answer[-4000:-2000]

if len(target) != len(answer):
    raise Exception

ratio = 0.9
border = int(len(target) * ratio)
print(len(target), border, len(target) - border)

# from tensorflow.keras.utils import to_categorical
# answer = to_categorical(answer)
# print(answer)

target_train = target[:border]
target_test  = target[border:]
answer_train = answer[:border]
answer_test  = answer[border:]

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

# model = tf.keras.models.load_model('./model')
# sys.exit()

from sklearn.svm import SVC

C_all = [1 + i * 0.4 for i in range(-1, 9)]

for C in C_all:
    print(f'C: {C}')

    model = SVC(C=C)
    model.fit(x_train, y_train)

    print(f'accuracy_train: {model.score(x_train, y_train)}')
    print(f'accuracy_test: {model.score(x_test, y_test)}')

    y_predict = model.predict(x_test)

    # print(len(y_test), len(y_predict))

    print(f'accuracy: {(np.count_nonzero(y_test == y_predict)) / len(y_predict) * 100:0.4f}(%)')
    for i in [0, 1, 2]:
        print(f'accuracy: {i}: {np.count_nonzero((y_test == y_predict) & (y_predict == i)) / len(y_predict) * 100:0.4f}(%)')

# print(y_predict)
# sys.exit()
