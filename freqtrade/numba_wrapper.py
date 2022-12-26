import numba

def _jit(*args, **kwargs):
    kwargs['nopython'] = True
    kwargs['cache'] = True

    if False:
        kwargs['inline'] = 'always'

    return numba.core.decorators.jit(*args, **kwargs)

numba.jit = _jit
numba.Dict = numba.typed.Dict
numba.List = numba.typed.List
