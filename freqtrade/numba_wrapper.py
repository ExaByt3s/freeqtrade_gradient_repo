import numba

def _jit(**kwargs):
    return numba.jit(nopython=True, cache=True, **kwargs)

def _njit(**kwargs):
    return numba.njit(cache=True, **kwargs)

# numba.jit = _jit
# numba.njit = _njit
numba.Dict = numba.typed.Dict
numba.List = numba.typed.List

'''
def njit(*args, **kws):
    """
    Equivalent to jit(nopython=True)

    See documentation for jit function/decorator for full description.
    """
    if 'nopython' in kws:
        warnings.warn('nopython is set for njit and is ignored', RuntimeWarning)
    if 'forceobj' in kws:
        warnings.warn('forceobj is set for njit and is ignored', RuntimeWarning)
        del kws['forceobj']
    kws.update({'nopython': True})
    return jit(*args, **kws)
'''
