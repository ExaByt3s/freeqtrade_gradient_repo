import numba
import numpy
from numpy import ndarray
import numba_option

@numba.njit(inline=numba_option.inline())
def reduce_mask(a: ndarray, mask: ndarray) -> ndarray:
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
