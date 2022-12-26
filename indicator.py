import math
from typing import Literal, Optional

import numpy
from numpy import ndarray

from numba_wrapper import numba
from tensorflow_wrapper import tensorflow

''' for test
import talib.abstract as ta
def ema(x: ndarray, window: int) -> ndarray:
    return numpy.array(ta.EMA(x, timeperiod=window))
'''

'''
@numba.jit()
def _generate_window(x: ndarray, window: int, x_mask: Optional[ndarray] = None) -> (ndarray, ndarray):

    if len(x) < window:
        raise Exception

    if window < 1:
        raise Exception

    if x_mask is None:
        x_mask = numpy.full(len(x), True)

    if x_mask is not None and len(x) != len(x_mask):
        raise Exception

    for i in range(len(x)):
        x_mask[i] &= ~numpy.isnan(x[i]).any()

    if len(x.shape) == 1:
        y_shape = (x.shape[0], window)
    elif len(x.shape) == 2:
        y_shape = (x.shape[0], window, x.shape[1])
    elif len(x.shape) == 3:
        y_shape = (x.shape[0], window, x.shape[1], x.shape[2])
    else:
        raise Exception

    y = numpy.empty(y_shape, dtype=x.dtype)
    y_mask = numpy.empty(len(x), dtype='bool')

    for i in range(len(x)):
        if i < window - 1 or not x_mask[i]:
            y[i] = numpy.full(len(x), numpy.nan)
            y_mask[i] = False

        else:
            y[i] = x[i + 1 - window:i + 1]
            y_mask[i] = ~numpy.isnan(x[i + 1 - window:i + 1]).any()

    return (y, y_mask)
'''

@tensorflow.jit()
def _generate_window(x: ndarray, window: int, x_mask: Optional[ndarray] = None) -> (ndarray, ndarray):

    if len(x) < window:
        raise Exception

    if window < 1:
        raise Exception

    if x_mask is None:
        x_mask = tensorflow.numpy.full(len(x), True)

    if x_mask is not None and len(x) != len(x_mask):
        raise Exception

    if len(x.shape) == 1:
        y_shape = [x.shape[0], window]
    elif len(x.shape) == 2:
        y_shape = [x.shape[0], window, x.shape[1]]
    elif len(x.shape) == 3:
        y_shape = [x.shape[0], window, x.shape[1], x.shape[2]]
    else:
        raise Exception

    x_mask = tensorflow.map_fn(fn=lambda i: x_mask[i] & tensorflow.numpy.any(~tensorflow.numpy.isnan(x[i])),
                               elems=tensorflow.numpy.arange(len(x)), fn_output_signature='bool')

    y = tensorflow.map_fn(fn=(lambda i: tensorflow.numpy.full(y_shape[1:], numpy.nan) if i < window - 1 or not x_mask[i]
                              else x[i + 1 - window:i + 1]),
                          elems=tensorflow.numpy.arange(len(x)), fn_output_signature=x.dtype)

    y_mask = tensorflow.map_fn(fn=(lambda i: False if i < window - 1 or not x_mask[i]
                                   else tensorflow.numpy.any(~tensorflow.numpy.isnan(x[i + 1 - window:i + 1]))),
                               elems=tensorflow.numpy.arange(len(x)), fn_output_signature='bool')

    return (y, y_mask)

@numba.jit
def ema(x: ndarray, window: int) -> ndarray:
    # if numpy.any(~numpy.isnan(x)):
        # raise Exception

    alpha = 2 / (1 + window)
    y = numpy.empty_like(x)
    y[0] = x[0]

    for i in range(1, len(x)):
        if not numpy.isnan(y[i - 1]):
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
        else:
            y[i] = x[i]

    for i in range(window - 1):
        y[i] = numpy.nan

    return y

@numba.jit
def ema_window(x: ndarray, window: int) -> ndarray:
    alpha = 2 / (1 + window)

    y = numpy.empty(len(x), dtype='float64')

    for i in range(window - 1):
        y[i] = numpy.nan

    for i in range(window - 1, len(x)):
        sum = x[i - (window - 1)]
        for j in range(1, window):
            sum = alpha * x[i - (window - 1) + j] + (1 - alpha) * sum

        y[i] = sum

    return y

@numba.jit
def zigzag_initial(input_a: ndarray, threshold_rate_up: float, threshold_rate_down: float) -> int:
    x_maximum = input_a[0]
    x_minimum = input_a[0]
    t_maximum = 0
    t_minimum = 0

    for t in range(1, len(input_a)):
        if input_a[t] / x_minimum > threshold_rate_up:
            if t_minimum == 0:
                return -1
            else:
                return 1

        if input_a[t] / x_maximum < threshold_rate_down:
            if t_maximum == 0:
                return 1
            else:
                return -1

        if input_a[t] > x_maximum:
            x_maximum = input_a[t]
            t_maximum = t

        if input_a[t] < x_minimum:
            x_minimum = input_a[t]
            t_minimum = t

    if input_a[0] < input_a[len(input_a) - 1]:
        return -1
    else:
        return 1


@numba.jit
def zigzag(input_a: ndarray, threshold_up: float, threshold_down: float) -> ndarray:
    result = numpy.empty(len(input_a), dtype='int64')

    for t in range(len(input_a)):
        result[t] = 0

    threshold_rate_up = 1 + threshold_up
    threshold_rate_down = 1 - threshold_down

    result_zigzag_initial = zigzag_initial(input_a, threshold_rate_up, threshold_rate_down)

    if True:
        result[0] = -result_zigzag_initial

    trend = -result_zigzag_initial
    x_pivot_last = input_a[0]

    for t in range(1, len(input_a)):
        if trend == -1:
            if input_a[t] / x_pivot_last > threshold_rate_up:
                result[t] = 1
                trend = 1
                x_pivot_last = input_a[t]

            elif input_a[t] < x_pivot_last:
                x_pivot_last = input_a[t]

        else:
            if input_a[t] / x_pivot_last < threshold_rate_down:
                result[t] = -1
                trend = -1
                x_pivot_last = input_a[t]

            elif input_a[t] > x_pivot_last:
                x_pivot_last = input_a[t]

    if True:
        if result[len(input_a) - 1] == 0:
            result[len(input_a) - 1] = trend

    return result


@numba.jit
def zigzag4(input_a: ndarray, threshold_up: float, threshold_down: float) -> ndarray:
    result = numpy.empty(len(input_a), dtype='int64')

    for t in range(len(input_a)):
        result[t] = 0

    threshold_rate_up = 1 + threshold_up
    threshold_rate_down = 1 - threshold_down

    trend = -(zigzag_initial(input_a, threshold_rate_up, threshold_rate_down))
    x_maximum_last = input_a[0]
    x_minimum_last = input_a[0]
    x_maximum_last_second = input_a[0]
    x_minimum_last_second = input_a[0]

    for t in range(1, len(input_a)):
        if trend == -1:
            if input_a[t] / x_minimum_last > threshold_rate_up:
                trend = 1

                if x_minimum_last > x_minimum_last_second:
                    result[t] = 1

                x_maximum_last_second = x_maximum_last
                x_maximum_last = input_a[t]

            elif input_a[t] < x_minimum_last:
                x_minimum_last = input_a[t]

        else:
            if input_a[t] / x_maximum_last < threshold_rate_down:
                trend = -1

                if x_maximum_last_second > x_maximum_last:
                    result[t] = -1

                x_minimum_last_second = x_minimum_last
                x_minimum_last = input_a[t]

            elif input_a[t] > x_maximum_last:
                x_maximum_last = input_a[t]

    return result


@numba.jit
def moving_average_simple(x: ndarray, window: int) -> ndarray:
    result = numpy.empty(len(x), dtype='float64')

    for i in range(window - 1):
        result[i] = math.nan

    for i in range(window - 1, len(x)):
        sum = 0
        for j in range(window):
            sum += x[i - j]

        result[i] = sum / window

    return result


@numba.jit
def window_sum_integer(x: ndarray, window: int) -> ndarray:
    result = numpy.empty(len(x), dtype='int64')

    for i in range(window - 1):
        result[i] = 0

    sum_initial = 0

    for i in range(window):
        sum_initial += x[(window - 1) - i]

    result[window - 1] = sum_initial

    for i in range(window, len(x)):
        result[i] = result[i - 1] + x[i] - x[i - window]

    return result


@numba.jit
def sum_range_integer(start: int, end: int) -> ndarray:
    height = start + (end - 1)
    width = (end - 1) - start + 1
    return height * width // 2


@numba.jit
def window_sum_range_integer(x: ndarray, window: int) -> ndarray:
    result = numpy.empty(len(x), dtype='int64')

    for i in range(window - 1):
        result[i] = 0

    for i in range(window - 1, len(x)):
        result[i] = sum_range_integer(x[i + 1 - window], x[i] + 1)

    return result


@numba.jit
def window_sum_float(x: ndarray, window: int) -> ndarray:
    result = numpy.empty(len(x), dtype='float64')

    for i in range(window - 1):
        result[i] = math.nan

    for i in range(window - 1, len(x)):
        sum = x[i]
        for j in range(1, window):
            sum += x[i - j]

        result[i] = sum

    return result


@numba.jit
def regression_1(y: ndarray, window: int) -> ndarray:
    if window <= 1:
        raise Exception

    x = numpy.arange(len(y))
    # sum_x = window_sum_integer(x, window)
    sum_x = window_sum_range_integer(x, window)
    sum_y = window_sum_float(y, window)
    sum_x2 = window_sum_integer(x ** 2, window)
    sum_xy = window_sum_float(x * y, window)
    slope = (window * sum_xy - sum_x * sum_y) / (window * sum_x2 - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / window
    return slope * x + intercept

# https://qiita.com/tkoba2/items/e0791bea345acb744195
@numba.jit
def rci(close: numpy.ndarray, timeperiod: int) -> numpy.ndarray:
    rci = numpy.full_like(close, numpy.nan)
    rank_period = numpy.arange(1, timeperiod + 1)
    for i in range(timeperiod - 1, len(close)):
        rank_price = close[i - timeperiod + 1:i + 1]
        rank_price = numpy.argsort(numpy.argsort(rank_price)) + 1
        aa = 6 * sum((rank_period - rank_price)**2)
        bb = timeperiod * (timeperiod**2 - 1)
        rci[i] = (1 - aa / bb) * 100
    return rci

def rci_v2(close: numpy.ndarray, timeperiod: int) -> numpy.ndarray:
    rank_target = [numpy.roll(close, i, axis=-1) for i in range(timeperiod)]
    rank_target = numpy.vstack(rank_target)[:, timeperiod - 1:]
    price_rank = numpy.argsort(numpy.argsort(rank_target[::-1], axis=0), axis=0) + 1
    time_rank = numpy.arange(1, timeperiod + 1).reshape(timeperiod, -1)
    aa = numpy.sum((time_rank - price_rank)**2, axis=0, dtype=float) * 6
    bb = float(timeperiod * (timeperiod**2 - 1))
    cc = numpy.divide(aa, bb, out=numpy.zeros_like(aa), where=bb != 0)
    rci = (1 - cc) * 100
    rci = numpy.concatenate([numpy.full(timeperiod - 1, numpy.nan), rci], axis=0)
    return rci

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
