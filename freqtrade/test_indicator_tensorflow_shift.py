import numpy
import indicator

x = numpy.arange(27, dtype=numpy.float64)
print(x)
y = indicator.tensorflow_shift(x, 2)
print(y)

x = numpy.arange(27, dtype=numpy.float64)
print(x)
y = indicator.tensorflow_shift(x, 0)
print(y)

x = numpy.arange(27, dtype=numpy.float64)
print(x)
y = indicator.tensorflow_shift(x, -2)
print(y)
