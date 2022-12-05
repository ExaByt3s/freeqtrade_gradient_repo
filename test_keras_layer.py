import tensorflow as tf

from keras_layer import RelativePosition

layer = RelativePosition()
out = layer(tf.Variable([tf.range(10), tf.range(10)]))
answer = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5,
                      1, 2, 3, 4, 1, 2, 3, 1, 2, 1])
print(out == answer)
if not tf.math.reduce_any(out == answer):
    print(f'failed: out:{out} answer:{answer}')
else:
    print(f'{out}')

'''
from keras_layer import DenseInputBias

layer = DenseInputBias()
out = layer(tf.Variable([tf.range(10), tf.range(10)]))
answer = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(out)
# print(tf.math.equal(out, answer))
# if not tf.math.reduce_any(tf.math.equal(out, answer)):
    # print(f'failed: out:{out} answer:{answer}')
# else:
    # print(f'{out}')
'''

from keras_layer import DenseAverage

layer = DenseAverage(10)
out = layer(tf.Variable([tf.range(10), tf.range(10)], dtype='float'))
answer = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5,
                      1, 2, 3, 4, 1, 2, 3, 1, 2, 1])
print(out)
# print(tf.math.equal(out, answer))
# if not tf.math.reduce_any(tf.math.equal(out, answer)):
    # print(f'failed: out:{out} answer:{answer}')
# else:
    # print(f'{out}')
