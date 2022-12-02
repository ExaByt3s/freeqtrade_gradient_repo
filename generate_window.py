import numpy
import numba
from numpy import ndarray
import numba_option

@numba.njit(inline=numba_option.inline())
def generate_window(a: ndarray, mask: ndarray, window: int = 200) -> (ndarray, ndarray):

    if window < 1:
        raise Exception

    if len(a) < window:
        raise Exception

    if len(a) != len(mask):
        raise Exception

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

    result = numpy.empty((length, window), dtype=a.dtype)
    index = 0

    for i in range(window - 1, len(a)):
        if mask_window[i]:
            result[index] = a[i + 1 - window:i + 1]
            index += 1

    return (result, mask_window)

@numba.njit(inline=numba_option.inline())
def generate_window_v2(a: ndarray, mask: ndarray, window: int = 200, exclude_nan: bool = False) -> (ndarray, ndarray):

    if window < 1:
        raise Exception

    if len(a) < window:
        raise Exception

    if len(a) != len(mask):
        raise Exception

    if exclude_nan:
        # if len(a.shape) == 1:
        if len(a.shape) == 2:
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

    # if len(a.shape) == 1:
        # result = numpy.empty((length, window), dtype=a.dtype)
    if len(a.shape) == 2:
        result = numpy.empty((length, window, a.shape[1]), dtype=a.dtype)

    index = 0

    for i in range(window - 1, len(a)):
        if mask_window[i]:
            result[index] = a[i + 1 - window:i + 1]
            index += 1

    return (result, mask_window)
