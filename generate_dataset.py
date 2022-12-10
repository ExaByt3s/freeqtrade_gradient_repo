import sys
from typing import Optional
import numba
import numpy
from numpy import ndarray
from pandas import DataFrame

def jit(**kwargs):
    return numba.njit(cache=True, **kwargs)

@jit(inline='always')
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

    return result_answer

@jit(inline='always')
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

@jit(inline='always')
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

@jit(inline='always')
def _make_divisible(number: int, divisor: int) -> int:
    return number - (number % divisor)

# @jit(inline='always')
def _window_nomalization(input_a: ndarray, threshold_scale: float = 0.005) -> ndarray:

    if len(input_a.shape) != 3:
        raise Exception

    window_maximum = numpy.amax(input_a, axis=1) * (1 + threshold_scale)
    window_minimum = numpy.amin(input_a, axis=1) * (1 - threshold_scale)
    window_maximum = numpy.repeat(window_maximum, input_a.shape[1], axis=0).reshape(input_a.shape)
    window_minimum = numpy.repeat(window_minimum, input_a.shape[1], axis=0).reshape(input_a.shape)
    result = (input_a - window_minimum) / (window_maximum - window_minimum)
    return result

# @jit()
def generate_dataset(input_info: ndarray, input_close: ndarray, input_mask: ndarray, window: int = 200,
                     threshold: float = 0.04, batch_size: int = 200, split_ratio: float = 0.8, train_include_test: bool = False,
                     enable_window_nomalization: bool = False) -> (ndarray, ndarray, ndarray, ndarray):  # , input_mask: Optional[ndarray] = None

    if len(input_info.shape) != 2:
        raise Exception

    if len(input_close.shape) != 1:
        raise Exception

    if len(input_mask.shape) != 1:
        raise Exception

    if not len(input_info) == len(input_close) == len(input_mask):
        raise Exception

    if window < 1:
        raise Exception

    if threshold < 0:
        raise Exception

    if batch_size < 2:
        raise Exception

    if not 0 <= split_ratio <= 1:
        raise Exception

    y = _generate_answer(input_close, threshold=threshold, enum_unknown=-1, enum_up=1, enum_down=0)
    y_mask = (y != -1)
    x, x_mask = _generate_window(input_info, (input_mask & y_mask), window=window, exclude_nan=True)
    y = _reduce_mask(y, x_mask)

    if len(x) != len(y):
        raise Exception

    # if enable_window_nomalization:
        # x = _window_nomalization(x, threshold_scale=0.005)

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
