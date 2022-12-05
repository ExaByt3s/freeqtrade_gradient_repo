# from technical import qtpylib
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.configuration import Configuration, TimeRange
from freqtrade.data.history import load_data
from freqtrade.enums import CandleType
from pathlib import Path

import generate_answer
import generate_window
import indicator
import reduce_mask

def generate_dataset():
    pass


pair_information = [
    'BTC/USDT',
    'ETH/USDT',
]

pair_trade = [
    # 'BTC/USDT',
    'ETH/USDT',
]

config = Configuration.from_files(['./freqtrade/config.json'])
data = load_data(
    datadir=Path('./freqtrade/user_data/data/binance'),
    pairs=pair_information,
    timeframe='5m',
    timerange=TimeRange.parse_timerange('20210801-20220101'),
    startup_candles=0,
    data_format=config['dataformat_ohlcv'],
    candle_type=CandleType.FUTURES,
)


def add_indicator(dataframe: DataFrame) -> DataFrame:
    dataframe['heikin-ashi_close'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
    dataframe['moving_average_simple_200'] = indicator.moving_average_simple(dataframe['heikin-ashi_close'].to_numpy(), window=200)
    dataframe['regression_1_200'] = indicator.regression_1(dataframe['heikin-ashi_close'].to_numpy(), window=200)
    dataframe['EMA200'] = ta.EMA(dataframe['heikin-ashi_close'], timeperiod=200)
    dataframe['WMA200'] = ta.WMA(dataframe['heikin-ashi_close'], timeperiod=200)
    # dataframe'RSI9'] = ta.RSI(dataframe['heikin-ashi_close'], timeperiod=9)
    dataframe['HMA200'] = qtpylib.hma(dataframe['heikin-ashi_close'], window=200)
    return dataframe

def add_information(dataframe_information: DataFrame, dataframe: DataFrame, pair: str) -> DataFrame:
    column = dataframe.columns.tolist()
    for i in column:

    return dataframe

dataframe_information = DataFrame()

for pair in pair_information:
    dataframe = data[pair]
    dataframe = add_indicator(dataframe)
    dataframe_information = add_information(dataframe_information, dataframe, pair)

dataframe = data['ETH/USDT']
dataframe = dataframe.drop(['date'], axis=1)
mask_volume = (dataframe['volume'] > 0)

window_backward = 200
answer = generate_answer.generate_answer_v2(dataframe['close'].to_numpy(), threshold=0.04, enum_unknown=-1,
                                            enum_up=1, enum_down=0)
mask_answer = (answer != -1)


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

target, mask_final = (
    generate_window.generate_window_v2(
        dataframe.to_numpy(), (mask_volume & mask_answer).to_numpy(), window=window_backward, exclude_nan=True
    )
)

answer = reduce_mask.reduce_mask(answer, mask_final)

if len(target) != len(answer):
    raise Exception

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


# def print_summary_y(name: str, a: ndarray):
    # for i in [0, 1, 2]:
        # print(f'{name}: {i}: {np.count_nonzero(a == i) / len(a) * 100:0.4f}(%)')
#
# print_summary_y('All of answer', answer)
# print_summary_y('Train part of answer', answer_train)
# print_summary_y('Test part of answer', answer_test)
