import sys
import numba
import numpy
from numpy import ndarray
from pandas import DataFrame
from generate_dataset import generate_dataset

if __name__ == '__main__':

    dataframe = DataFrame(data={
        'col1': [1, 2],
        'col2': [3, 4],
    })

    print(dataframe.to_markdown())

    # sys.exit()

# from technical import qtpylib
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.configuration import Configuration, TimeRange
from freqtrade.data.history import load_data
from freqtrade.enums import CandleType
from pathlib import Path
import indicator


pair_information = [
    # 'BTC/USDT',
    'ETH/USDT',
]
#
# pair_trade = [
    # # 'BTC/USDT',
    # 'ETH/USDT',
# ]
#
# # config = Configuration.from_files(['./freqtrade/config.json'])
data = load_data(
    datadir=Path('./freqtrade/user_data/data/binance'),
    pairs=pair_information,
    timeframe='5m',
    timerange=TimeRange.parse_timerange('20210801-20220101'),
    startup_candles=0,
    # data_format=config['dataformat_ohlcv'],
    data_format='jsongz',
    candle_type=CandleType.FUTURES,
)

dataframe = data['ETH/USDT']
dataframe = dataframe.drop(['date'], axis=1)

x_train, y_train, x_test, y_test = generate_dataset(dataframe.to_numpy(), dataframe['close'].to_numpy(),
                                                    (dataframe['volume'] > 0).to_numpy(), window=200, threshold=0.04,
                                                    batch_size=200, split_ratio=0.8)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))

sys.exit()

def add_indicator(dataframe: DataFrame) -> DataFrame:
    dataframe['heikin-ashi_close'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
    dataframe['moving_average_simple_200'] = indicator.moving_average_simple(dataframe['heikin-ashi_close'].to_numpy(), window=200)
    dataframe['regression_1_200'] = indicator.regression_1(dataframe['heikin-ashi_close'].to_numpy(), window=200)
    dataframe['EMA200'] = ta.EMA(dataframe['heikin-ashi_close'], timeperiod=200)
    dataframe['WMA200'] = ta.WMA(dataframe['heikin-ashi_close'], timeperiod=200)
    # dataframe'RSI9'] = ta.RSI(dataframe['heikin-ashi_close'], timeperiod=9)
    dataframe['HMA200'] = qtpylib.hma(dataframe['heikin-ashi_close'], window=200)
    return dataframe
#
# def add_information(dataframe_information: DataFrame, dataframe: DataFrame, pair: str) -> DataFrame:
    # column = dataframe.columns.tolist()
    # for i in column:
#
    # return dataframe
#
# dataframe_information = DataFrame()
#
# for pair in pair_information:
    # dataframe = data[pair]
    # dataframe = add_indicator(dataframe)
    # dataframe_information = add_information(dataframe_information, dataframe, pair)

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

# print(target[:20], answer[:20])
# sys.exit()

# ratio = 0.7
# border = int(len(target) * ratio)
length_margin = 1600
length_test = 3200
length_train = 10000
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

dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

dataset_train = dataset_train.batch(batch_size, drop_remainder=True)
dataset_test = dataset_test.batch(batch_size, drop_remainder=True)

# option = tf.data.Options()
# option.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
# data_train = data_train.with_options(option)
# data_test = data_test.with_options(option)


# def print_summary_y(name: str, a: ndarray):
    # for i in [0, 1, 2]:
        # print(f'{name}: {i}: {np.count_nonzero(a == i) / len(a) * 100:0.4f}(%)')
#
# print_summary_y('All of answer', answer)
# print_summary_y('Train part of answer', answer_train)
# print_summary_y('Test part of answer', answer_test)
