import numpy
import numba
from numpy import ndarray
import numba_option

@numba.njit(inline=numba_option.inline())
def generate_answer(price: ndarray, window_backward: int = 200, window_forward: int = 200) -> ndarray:
    '''
    profit: float = 0.10
    loss: float = 0.10
    enum_nan: int = -1
    enum_nothing: int = 0
    enum_long: int = 1
    enum_short: int = 2

    if window_backward < 1:
        raise Exception

    if window_forward < 1:
        raise Exception

    if len(price) < window_backward + window_forward:
        raise Exception

    result_answer = numpy.empty(len(price), dtype='int64')

    for i in range(window_backward - 1):
        result_answer[i] = enum_nan

    for i in range(len(price) - window_forward, len(price)):
        result_answer[i] = enum_nan

    for i in range(window_backward - 1, len(price) - window_forward):
        price_entry = price[i]
        answer = enum_nothing

        for j in range(1, window_forward + 1):
            price_current = price[i + j]
            # print(price_entry, price_current, price_current / price_entry)

            if price_current / price_entry > (1 + profit):
                answer = enum_long
                break

            elif price_current / price_entry < (1 - loss):
                answer = enum_short
                break

        result_answer[i] = answer
    '''
    profit_long: float = 0.01
    loss_long: float = 0.01
    profit_short: float = 0.01
    loss_short: float = 0.01
    enum_nan: int = -1
    enum_nothing: int = 0
    enum_long: int = 1
    enum_short: int = 0

    if window_backward < 1:
        raise Exception

    if window_forward < 1:
        raise Exception

    if len(price) < window_backward + window_forward:
        raise Exception

    result_answer = numpy.empty(len(price), dtype='int64')

    for i in range(window_backward - 1):
        result_answer[i] = enum_nan

    for i in range(len(price) - window_forward, len(price)):
        result_answer[i] = enum_nan

    for i in range(window_backward - 1, len(price) - window_forward):
        price_entry = price[i]
        answer = enum_nothing

        for j in range(1, window_forward + 1):
            price_current = price[i + j]

            if price_current / price_entry > (1 + profit_long):
                answer = enum_long
                break

            elif price_current / price_entry < (1 - loss_long):
                break

        if answer == enum_long:
            result_answer[i] = answer
            continue

        for j in range(1, window_forward + 1):
            price_current = price[i + j]

            if price_entry / price_current > (1 + profit_short):
                answer = enum_short
                break

            elif price_entry / price_current < (1 - loss_short):
                break

        result_answer[i] = answer

    return result_answer

@numba.njit(inline=numba_option.inline())
def generate_answer_v2(price: ndarray, threshold: float = 0.01, enum_unknown: int = -1, enum_up: int = 1, enum_down: int = 0) -> ndarray:

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
