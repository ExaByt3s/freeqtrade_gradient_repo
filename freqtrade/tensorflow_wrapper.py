import tensorflow

def _tensorflow_jit(**kwargs):
    return tensorflow.function(jit_compile=True, **kwargs)

tensorflow.jit = _tensorflow_jit
tensorflow.numpy = tensorflow.experimental.numpy

if False:
    tensorflow.numpy.experimental_enable_numpy_behavior()

def _nan_to_num(x, nan=0., posinf=None, neginf=None):
    posinf = tensorflow.numpy.inf
    neginf = -tensorflow.numpy.inf
    x = tensorflow.numpy.where(tensorflow.numpy.isnan(x), nan, x)
    x = tensorflow.numpy.where(tensorflow.numpy.isposinf(x), posinf, x)
    x = tensorflow.numpy.where(tensorflow.numpy.isneginf(x), neginf, x)
    return x

tensorflow.numpy.nan_to_num = _nan_to_num
