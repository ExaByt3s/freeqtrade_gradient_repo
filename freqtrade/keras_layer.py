import tensorflow as tf
from tensorflow.keras.layers import Layer
jit = tf.function(jit_compile=True)

class RelativePosition(Layer):
    def __init__(self):
        super(RelativePosition, self).__init__()

    def build(self, input_shape):
        n = input_shape[1]
        x, y = tf.meshgrid(tf.range(n), tf.range(n))
        self.mask = tf.Variable(initial_value=tf.math.greater(x, y), trainable=False)

    # @jit
    def _f(self, a):
        x, y = tf.meshgrid(a, a)
        x = tf.math.subtract(x, y)
        x = tf.boolean_mask(x, self.mask)
        # x = tf.ensure_shape(x, [self.n * (self.n - 1) // 2])
        return x

    # @jit
    def call(self, inputs):
        x = tf.map_fn(fn=self._f, elems=inputs)
        n = inputs.shape[1]
        x = tf.ensure_shape(x, [inputs.shape[0], n * (n - 1) // 2])
        return x

@jit
def relative_position(inputs):
    n = inputs.shape[-1]
    x_mask, y_mask = tf.meshgrid(tf.range(n), tf.range(n))
    mask = tf.math.greater(x_mask, y_mask)
    x, y = tf.meshgrid(inputs, inputs)
    x = tf.math.subtract(x, y)
    x = tf.boolean_mask(x, mask)
    x = tf.ensure_shape(x, [n * (n - 1) // 2])
    return x

class SimpleDense(Layer):
    def __init__(self, units=32):
        super(SimpleDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='b', shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class DenseNotTainable(Layer):
    def __init__(self, units):
        super(DenseNotTainable, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=False)
        self.b = self.add_weight(name='b', shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=False)
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class DenseAverage(Layer):
    def __init__(self, units=32):
        super(DenseAverage, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='b', shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
    def call(self, inputs):
        x = tf.matmul(inputs, self.w)
        s = tf.reduce_sum(self.w, axis=0)
        x /= s
        x += self.b
        return x

from tensorflow.keras.layers import BatchNormalization

class DenseBatchNormalization(Layer):
    def __init__(self, units=32):
        super(DenseBatchNormalization, self).__init__()
        self.units = units
        self.bn = BatchNormalization()

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='b', shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
    def call(self, inputs):
        w2 = self.bn(self.w)
        x = tf.matmul(inputs, w2)
        x += self.b
        return x

'''
class DenseInputBias(Layer):
  def __init__(self,
               # units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(DenseInputBias, self).__init__(
        activity_regularizer=activity_regularizer, **kwargs)

    # self.units = int(units) if not isinstance(units, int) else units
    # if self.units < 0:
      # raise ValueError(f'Received an invalid value for `units`, expected '
                       # f'a positive integer, got {units}.')
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.input_spec = InputSpec(min_ndim=2)
    self.supports_masking = True

  def build(self, input_shape):
    self.units = input_shape[1]

    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `DenseInputBias` layer with non-floating point '
                      'dtype %s' % (dtype,))

    input_shape = tensor_shape.TensorShape(input_shape)
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `DenseInputBias` '
                       'should be defined. Found `None`.')

    self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
    self.kernel = self.add_weight(
        'kernel',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
      inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)

    rank = inputs.shape.rank
    # if rank == 2 or rank is None:
    if False:
      # We use embedding_lookup_sparse as a more efficient matmul operation for
      # large sparse input tensors. The op will result in a sparse gradient, as
      # opposed to sparse_ops.sparse_tensor_dense_matmul which results in dense
      # gradients. This can lead to sigfinicant speedups, see b/171762937.
      if isinstance(inputs, sparse_tensor.SparseTensor):
        # We need to fill empty rows, as the op assumes at least one id per row.
        inputs, _ = sparse_ops.sparse_fill_empty_rows(inputs, 0)
        # We need to do some munging of our input to use the embedding lookup as
        # a matrix multiply. We split our input matrix into separate ids and
        # weights tensors. The values of the ids tensor should be the column
        # indices of our input matrix and the values of the weights tensor
        # can continue to the actual matrix weights.
        # The column arrangement of ids and weights
        # will be summed over and does not matter. See the documentation for
        # sparse_ops.sparse_tensor_dense_matmul a more detailed explanation
        # of the inputs to both ops.
        ids = sparse_tensor.SparseTensor(
            indices=inputs.indices,
            values=inputs.indices[:, 1],
            dense_shape=inputs.dense_shape)
        weights = inputs
        outputs = embedding_ops.embedding_lookup_sparse_v2(
            self.kernel, ids, weights, combiner='sum')
      else:
        # outputs = gen_math_ops.MatMul(a=inputs, b=self.kernel)
        outputs = tf.linalg.matmul(a=inputs, b=self.kernel)
    # Broadcast kernel to inputs.
    else:
      outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.kernel.shape[-1]]
        outputs.set_shape(output_shape)

    if self.use_bias:
      outputs = nn_ops.bias_add(outputs, self.bias)

    outputs = tf.tensordot(inputs, self.kernel, axes=1)
    outputs += self.bias
    outputs += inputs

    # if self.activation is not None:
      # outputs = self.activation(outputs)

    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % (input_shape,))
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = super(DenseInputBias, self).get_config()
    config.update({
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    })
    return config
'''

'''
class DenseAverage(Layer):
    def __init__(self, n, activity_regularizer=None,
        kernel_initializer='glorot_uniform', kernel_regularizer=None, kernel_constraint=None,
        bias_initializer='zeros', bias_regularizer=None, bias_constraint=None):

        super(DenseAverage, self).__init__(activity_regularizer=activity_regularizer)
        self.n = n
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):

        self.kernel = self.add_weight(
            'kernel',
            shape=[input_shape[1], self.n],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)

        self.bias = self.add_weight(
            'bias',
            shape=[self.n,],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=self.dtype,
            trainable=True)

    def call(self, inputs):
        outputs = tf.tensordot(inputs, self.kernel, axes=1)
        outputs += self.bias
        return outputs

    def get_config(self):
        config = super(DenseAverage, self).get_config()
        config.update({
            'n': self.n,
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        })
        return config
'''

'''
class SimpleDense(Layer):

  def __init__(self, units=32):
      super(SimpleDense, self).__init__()
      self.units = units

  def build(self, input_shape):  # Create the state of the layer (weights)
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(
        initial_value=w_init(shape=(input_shape[-1], self.units),
                             dtype='float32'),
        trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(
        initial_value=b_init(shape=(self.units,), dtype='float32'),
        trainable=True)

  def call(self, inputs):  # Defines the computation from inputs to outputs
      return tf.matmul(inputs, self.w) + self.b
'''
