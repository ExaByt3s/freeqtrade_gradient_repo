import sys
from typing import Optional
import numpy
import tensorflow
from numpy import ndarray
from pandas import DataFrame

from numba_wrapper import numba
from tensorflow_wrapper import tensorflow

@numba.jit(inline='always')
def _generate_answer(price: ndarray, threshold: float = 0.01, enum_unknown: int = -1, enum_up: int = 1,
                     enum_down: int = 0) -> ndarray:

    result_answer = numpy.empty(len(price), dtype='int64')

    for i in range(len(price)):
        price_entry = price[i]
        answer = enum_unknown

        for j in range(1, len(price)):
            if i + j > len(price) - 1 :
                break

            price_current = price[i + j]

            if price_current / price_entry > (1 + threshold):
                answer = enum_up
                break

            elif price_current / price_entry < (1 - threshold):
                answer = enum_down
                break

        result_answer[i] = answer

        if answer == enum_unknown:
            for j in range(i, len(price)):
                result_answer[j] = enum_unknown
            break

    return (result_answer, result_answer != -1)

'''
@numba.jit
def shift(input_a: ndarray, period: int, fill_value: (int|float|str|bool)) -> ndarray:
    result = numpy.empty_like(input_a)

    for i in range(len(input_a) - period, len(input_a)):
        result[i] = fill_value

    for i in range(len(input_a) - period):
        result[i] = input_a[i + period]

    return result
'''

'''
# https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
@numba.jit
def shift(input_a: ndarray, period: int, fill_value: (int|float|str|bool) = numpy.nan) -> ndarray:
    result = numpy.empty_like(input_a)

    if period > 0:
        result[:period] = fill_value
        result[period:] = input_a[:-period]

    elif period < 0:
        result[period:] = fill_value
        result[:period] = input_a[-period:]

    else:
        result[:] = input_a

    return result
'''

import indicator

@numba.jit(inline='always')
def _generate_answer_sma(price: ndarray, window: int = 100, period_shift: int = -100) -> ndarray:
    if period_shift >= 0:
        raise Exception

    sma = indicator.moving_average_simple(price, window)
    sma_shift = shift(sma, period=period_shift, fill_value=numpy.nan)
    result = sma_shift / sma
    mask = ~numpy.isnan(result)
    return (result, mask)

def _generate_answer_ema(price: ndarray, window: int = 100, period_shift: int = -100) -> ndarray:
    if period_shift >= 0:
        raise Exception

    # ema = indicator.ema(price, window)
    ema = indicator.ema_window(price, window)
    ema_shift = shift(ema, period=period_shift, fill_value=numpy.nan)
    result = ema_shift / ema
    mask = ~numpy.isnan(result)
    return (result, mask)

def _generate_answer_regression1(price: ndarray, window: int = 100, period_shift: int = -100) -> ndarray:
    if period_shift >= 0:
        raise Exception

    y = indicator.regression_1(price, window)
    y_shift = shift(y, period=period_shift)
    result = y_shift / y
    mask = ~numpy.isnan(result)
    return (result, mask)

@numba.jit(inline='always')
def _generate_window(a: ndarray, mask: ndarray, window: int = 200, exclude_nan: bool = False) -> (ndarray, ndarray):

    if len(a) != len(mask):
        raise Exception

    if len(a) < window:
        raise Exception

    if len(a.shape) != 2:
        raise Exception

    if window < 1:
        raise Exception

    if exclude_nan:
        for i in range(len(a)):
            mask[i] &= ~numpy.isnan(a[i]).any()

    length: int = 0
    mask_window = numpy.empty(len(a), dtype='bool')

    for i in range(window - 1):
        mask_window[i] = False

    for i in range(window - 1, len(a)):
        status = True

        for j in range(window):
            if not mask[i - j]:
                status = False
                break

        if status:
            mask_window[i] = True
            length += 1
        else:
            mask_window[i] = False

    result = numpy.empty((length, window, a.shape[1]), dtype=a.dtype)
    index = 0

    for i in range(window - 1, len(a)):
        if mask_window[i]:
            result[index] = a[i + 1 - window:i + 1]
            index += 1

    return (result, mask_window)

'''
@numba.jit(inline='always')
def _reduce_mask(a: ndarray, mask: ndarray) -> ndarray:
    if a.shape != mask.shape:
        raise Exception

    if len(a.shape) != 1:
        raise Exception

    length = 0

    for i in range(len(a)):
        if mask[i]:
            length += 1

    result = numpy.empty(length, dtype=a.dtype)
    index: int = 0

    for i in range(len(a)):
        if mask[i]:
            result[index] = a[i]
            index += 1

    return result
'''

@numba.jit(inline='always')
def _make_divisible(number: int, divisor: int) -> int:
    return number - (number % divisor)

@tensorflow.jit()
def _window_nomalization(input_a: ndarray, threshold_scale: Optional[float]) -> ndarray:

    if len(input_a.shape) != 3:
        raise Exception

    window_maximum = tensorflow.numpy.amax(input_a, axis=1)
    if threshold_scale is not None:
        window_maximum *= (1 + threshold_scale)
    window_maximum = tensorflow.numpy.repeat(window_maximum, input_a.shape[1], axis=0)
    window_maximum = tensorflow.numpy.reshape(window_maximum, input_a.shape)

    window_minimum = tensorflow.numpy.amin(input_a, axis=1)
    if threshold_scale is not None:
        window_minimum *= (1 - threshold_scale)
    window_minimum = tensorflow.numpy.repeat(window_minimum, input_a.shape[1], axis=0)
    window_minimum = tensorflow.numpy.reshape(window_minimum, input_a.shape)

    result = (input_a - window_minimum) / (window_maximum - window_minimum)
    return result

def _window_nomalization_v2(input_a: ndarray) -> ndarray:

    if len(input_a.shape) != 3:
        raise Exception

    window_index0 = numpy.repeat(input_a[:, 0], input_a.shape[1], axis=0).reshape(input_a.shape)
    result = input_a / window_index0 - 1
    return result


def _window_nomalization_zscore(x: ndarray, axis: int = None):
    x_mean = x.mean(axis=axis, keepdims=True)
    x_std  = numpy.std(x, axis=axis, keepdims=True)
    zscore = (x - x_mean) / x_std
    return zscore

def generate_dataset(x: ndarray, x_mask: ndarray, y: ndarray, y_mask: ndarray, window: int, batch_size: int = 200,
                     split_ratio: float = 0.8, train_include_test: bool = False, enable_window_nomalization: bool = False,
                     encode_onehot: bool = False) -> (ndarray, ndarray, ndarray, ndarray):

    if len(x.shape) != 2:
        raise Exception

    if len(x_mask.shape) != 1:
        raise Exception

    if len(y.shape) != 2:
        raise Exception

    if len(y_mask.shape) != 1:
        raise Exception

    if not len(x) == len(x_mask) == len(y) == len(y_mask):
        raise Exception

    if window < 1:
        raise Exception

    if batch_size < 2:
        raise Exception

    if not 0 <= split_ratio <= 1:
        raise Exception

    # y, y_mask = _generate_answer(input_close, threshold=threshold, enum_unknown=-1, enum_up=1, enum_down=0)
    # y, y_mask = _generate_answer_sma(input_close, window=100, period_shift=-100)
    # y, y_mask = _generate_answer_ema(input_close, window=200, period_shift=-200)
    # y, y_mask = _generate_answer_regression1(input_close, window=100, period_shift=-100)

    y_mask &= ~numpy.isnan(y).any(axis=1)
    x, x_mask = _generate_window(x, (x_mask & y_mask), window=window, exclude_nan=True)
    y = y[x_mask]

    if len(x) != len(y):
        raise Exception

    if enable_window_nomalization and window >= 3:
        if True:
            x = _window_nomalization(x, threshold_scale=0.005)
        else:
            x = _window_nomalization_v2(x)

    if encode_onehot:
        y = tensorflow.keras.utils.to_categorical(y)

    length = _make_divisible(len(x), batch_size)
    x, y = x[-length:], y[-length:]

    # length_train = _make_divisible(length * split_ratio, batch_size)
    length_train = int(_make_divisible(length * split_ratio, batch_size))

    if train_include_test:
        x_train, y_train = x[:], y[:]
        x_test, y_test = x[length_train:], y[length_train:]

    else:
        x_train, y_train = x[:length_train], y[:length_train]
        x_test, y_test = x[length_train:], y[length_train:]

    return (x_train, y_train, x_test, y_test)

def generate_dataset_predict(x: ndarray, x_mask: ndarray, window: int, batch_size: int = 200,
                             enable_window_nomalization: bool = False) -> (ndarray, ndarray):

    if len(x.shape) != 2:
        raise Exception

    if len(x_mask.shape) != 1:
        raise Exception

    if not len(x) == len(x_mask):
        raise Exception

    if window < 1:
        raise Exception

    if batch_size < 2:
        raise Exception

    x, x_mask = _generate_window(x, x_mask, window=window, exclude_nan=True)

    if enable_window_nomalization and window >= 3:
        if True:
            x = _window_nomalization(x, threshold_scale=0.005)
        else:
            x = _window_nomalization_v2(x)

    return (x, x_mask)
