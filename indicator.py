import math
from typing import Literal, Optional

import numpy
import numba
from numpy import ndarray

import numba_option


@numba.njit(inline=numba_option.inline())
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


@numba.njit(inline=numba_option.inline())
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


@numba.njit(inline=numba_option.inline())
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


@numba.njit(inline=numba_option.inline())
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


@numba.njit(inline=numba_option.inline())
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


@numba.njit(inline=numba_option.inline())
def sum_range_integer(start: int, end: int) -> ndarray:
    height = start + (end - 1)
    width = (end - 1) - start + 1
    return height * width // 2


@numba.njit(inline=numba_option.inline())
def window_sum_range_integer(x: ndarray, window: int) -> ndarray:
    result = numpy.empty(len(x), dtype='int64')

    for i in range(window - 1):
        result[i] = 0

    for i in range(window - 1, len(x)):
        result[i] = sum_range_integer(x[i + 1 - window], x[i] + 1)

    return result


@numba.njit(inline=numba_option.inline())
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


@numba.njit(inline=numba_option.inline())
def regression_1(y: ndarray, window: int) -> ndarray:
    if window == 1:
        return y

    x = numpy.arange(len(y))
    # sum_x = window_sum_integer(x, window)
    sum_x = window_sum_range_integer(x, window)
    sum_y = window_sum_float(y, window)
    sum_x2 = window_sum_integer(x ** 2, window)
    sum_xy = window_sum_float(x * y, window)
    slope = (window * sum_xy - sum_x * sum_y) / (window * sum_x2 - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / window
    return slope * x + intercept
