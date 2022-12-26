import os
import pickle
import sys

from keras_device import scope
from tensorflow_wrapper import tensorflow

# @tensorflow.jit()
def loss_custom(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor) -> tensorflow.Tensor:
    batch_size = y_pred.shape[0]
    y_true = tensorflow.cast(y_true, y_pred.dtype)

    # print(y_pred)
    # print(y_true)

    if y_pred.shape != (batch_size, 3):
        raise Exception(y_pred.shape)
    if y_true.shape != (batch_size, 2):
        raise Exception(y_true.shape)

    # y_pred_index_maximum = tensorflow.numpy.argmax(y_pred, axis=1)
    # y_true_index_maximum = tensorflow.numpy.argmax(y_true, axis=1)

    # print(y_pred_index_maximum)
    # print(y_true_index_maximum)

    point_zero = y_pred[:, 2:]

    # print(point_zero)

    y_pred = y_pred[:, :2]

    # print(y_pred)

    y_difference = y_true - y_pred

    # print(y_difference)

    # mask_zero = tensorflow.numpy.repeat((y_pred_index_maximum == 2), 2).reshape(batch_size, 2)

    # print(mask_zero)

    # point_plus = y_pred[~mask_zero & (y_difference > 0)]
    # point_minus = -y_pred[mask_zero | (y_difference < 0)]
    point_plus = y_pred[y_difference > 0]
    point_minus = y_pred[y_difference < 0]

    # print(point_plus)
    # print(point_minus)

    fee = 0.002

    # loss = batch_size - tensorflow.numpy.sum(point_plus) + tensorflow.numpy.sum(point_minus) * 1.05
    # loss = batch_size - tensorflow.numpy.sum(point_plus) + tensorflow.numpy.sum(point_minus) - tensorflow.numpy.sum(point_zero) * 0
    loss = batch_size - (tensorflow.numpy.sum(point_plus) * (1 - fee)) + (tensorflow.numpy.sum(point_minus) * (1 + fee)) - tensorflow.numpy.sum(point_zero) * 0
    return loss

print(loss_custom(tensorflow.numpy.array([[1, 0], [1, 0], [1, 0]]), tensorflow.numpy.array([[0.1, 0.3, 0.6], [0.1, 0.6, 0.3], [0.6, 0.3, 0.1]])))
# assert loss_custom(tensorflow.numpy.array([[1, 0], [1, 0], [1, 0]]), tensorflow.numpy.array([[0.1, 0.3, 0.6], [0.1, 0.6, 0.3], [0.6, 0.3, 0.1]])) == 3.4

def reverse_direction(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor) -> tensorflow.Tensor:
    batch_size = y_pred.shape[0]
    y_true = tensorflow.cast(y_true, y_pred.dtype)

    if y_pred.shape != (batch_size, 3):
        raise Exception(y_pred.shape)
    if y_true.shape != (batch_size, 2):
        raise Exception(y_true.shape)

    y_pred_index_maximum = tensorflow.numpy.argmax(y_pred, axis=1)
    y_true_index_maximum = tensorflow.numpy.argmax(y_true, axis=1)

    # mask0 = (y_true_index_maximum == y_pred_index_maximum) & (y_true_index_maximum == 0)
    # ratio0 = tensorflow.numpy.count_nonzero(mask0) / len(y_true_index_maximum)

    mask0 = (y_true_index_maximum != y_pred_index_maximum) & (y_pred_index_maximum != 2)
    ratio0 = tensorflow.numpy.count_nonzero(mask0) / len(y_true_index_maximum)
    return ratio0

print(reverse_direction(tensorflow.numpy.array([[1, 0], [1, 0], [1, 0]]), tensorflow.numpy.array([[0.1, 0.3, 0.6], [0.1, 0.6, 0.3], [0.6, 0.3, 0.1]])))

def ratio2(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor) -> tensorflow.Tensor:
    batch_size = y_pred.shape[0]
    y_true = tensorflow.cast(y_true, y_pred.dtype)

    if y_pred.shape != (batch_size, 3):
        raise Exception(y_pred.shape)
    if y_true.shape != (batch_size, 2):
        raise Exception(y_true.shape)

    y_pred_index_maximum = tensorflow.numpy.argmax(y_pred, axis=1)
    y_true_index_maximum = tensorflow.numpy.argmax(y_true, axis=1)
    mask = y_pred_index_maximum == 2
    ratio = tensorflow.numpy.count_nonzero(mask) / len(y_true_index_maximum)
    return ratio

@tensorflow.jit()
def accuracy60(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor) -> tensorflow.Tensor:
    y_true = tensorflow.cast(y_true, y_pred.dtype)

    y_pred_index_maximum = tensorflow.numpy.argmax(y_pred, axis=1)
    y_true_index_maximum = tensorflow.numpy.argmax(y_true, axis=1)
    y_pred_maximum = tensorflow.numpy.amax(y_pred, axis=1)

    mask = (y_pred_index_maximum == y_true_index_maximum) & (y_pred_maximum > 0.60)
    ratio = tensorflow.numpy.count_nonzero(mask) / len(y_true_index_maximum)
    return ratio

print(accuracy60(tensorflow.numpy.array([[1, 0], [1, 0], [1, 0]]), tensorflow.numpy.array([[0.1, 0.3, 0.95], [0.1, 0.95, 0.3], [0.95, 0.3, 0.1]])))

@tensorflow.jit()
def reverse_direction60(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor) -> tensorflow.Tensor:
    y_true = tensorflow.cast(y_true, y_pred.dtype)

    y_pred_index_maximum = tensorflow.numpy.argmax(y_pred, axis=1)
    y_true_index_maximum = tensorflow.numpy.argmax(y_true, axis=1)
    y_pred_maximum = tensorflow.numpy.amax(y_pred, axis=1)

    mask = (y_true_index_maximum != y_pred_index_maximum) & (y_pred_maximum > 0.60)
    ratio = tensorflow.numpy.count_nonzero(mask) / len(y_true_index_maximum)
    return ratio

@tensorflow.jit()
def error_absolute_maximum(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor) -> tensorflow.Tensor:
    result = tensorflow.numpy.amax(tensorflow.numpy.absolute(y_true - y_pred))
    return result

_rate_entry = 0.04
_rate_exit_profit = 0.02
_rate_exit_loss = 0.001

if _rate_exit_profit > _rate_entry or _rate_exit_loss > _rate_entry:
    raise Exception

if _rate_entry <= 0 or _rate_exit_profit <= 0 or _rate_exit_loss <= 0 :
    raise Exception

@tensorflow.jit()
def ratio_entry(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor, rate_entry: float = _rate_entry) -> tensorflow.Tensor:
    mask = (y_pred > (1 + rate_entry)) | (y_pred < (1 - rate_entry))
    result = tensorflow.numpy.count_nonzero(mask) / len(y_pred)
    return result

@tensorflow.jit()
def ratio_exit_profit(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor, rate_entry: float = _rate_entry,
                      rate_exit_profit: float = _rate_exit_profit) -> tensorflow.Tensor:
    mask_a = (((y_pred > (1 + rate_entry)) & (y_true > (1 + rate_exit_profit)))
              | ((y_pred < (1 - rate_entry)) & (y_true < (1 - rate_exit_profit))))
    mask_b = (y_pred > (1 + rate_entry)) | (y_pred < (1 - rate_entry))
    result = tensorflow.numpy.count_nonzero(mask_a) / tensorflow.numpy.count_nonzero(mask_b)
    result = tensorflow.numpy.nan_to_num(result)
    return result

@tensorflow.jit()
def ratio_hold(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor, rate_entry: float = _rate_entry,
               rate_exit_profit: float = _rate_exit_profit, rate_exit_loss: float = _rate_exit_loss) -> tensorflow.Tensor:
    mask_a = (((y_pred > (1 + rate_entry)) & ((1 - rate_exit_loss) < y_true) & (y_true < (1 + rate_exit_profit)))
              | ((y_pred < (1 - rate_entry)) & ((1 + rate_exit_loss) > y_true) & (y_true > (1 - rate_exit_profit))))
    mask_b = (y_pred > (1 + rate_entry)) | (y_pred < (1 - rate_entry))
    result = tensorflow.numpy.count_nonzero(mask_a) / tensorflow.numpy.count_nonzero(mask_b)
    result = tensorflow.numpy.nan_to_num(result)
    return result

@tensorflow.jit()
def ratio_exit_loss(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor, rate_entry: float = _rate_entry,
                    rate_exit_loss: float = _rate_exit_loss) -> tensorflow.Tensor:
    mask_a = (((y_pred > (1 + rate_entry)) & (y_true < (1 - rate_exit_loss)))
              | ((y_pred < (1 - rate_entry)) & (y_true > (1 + rate_exit_loss))))
    mask_b = (y_pred > (1 + rate_entry)) | (y_pred < (1 - rate_entry))
    result = tensorflow.numpy.count_nonzero(mask_a) / tensorflow.numpy.count_nonzero(mask_b)
    result = tensorflow.numpy.nan_to_num(result)
    return result

@tensorflow.jit()
def percent(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor, rate_entry: float = _rate_entry) -> tensorflow.Tensor:
    mask_a = y_pred > (1 + rate_entry)
    mask_b = y_pred < (1 - rate_entry)

    sum_long = tensorflow.numpy.sum(y_true[mask_a] - 1)
    sum_short = tensorflow.numpy.sum(-(y_true[mask_b] - 1))
    result = (sum_long + sum_short) * 100
    return result

@tensorflow.jit()
def percent_loss(y_true: tensorflow.Tensor, y_pred: tensorflow.Tensor, rate_entry: float = _rate_entry) -> tensorflow.Tensor:
    mask_a = (y_pred > (1 + rate_entry)) & (y_true < 1)
    mask_b = (y_pred < (1 - rate_entry)) & (y_true > 1)

    sum_long = tensorflow.numpy.sum(y_true[mask_a] - 1)
    sum_short = tensorflow.numpy.sum(-(y_true[mask_b] - 1))
    result = (sum_long + sum_short) * 100
    return result

enable_cache = True
cachefile = 'cache.pickle'

if enable_cache and os.path.exists(cachefile):
    with open(cachefile, 'rb') as cachehandle:
        print(f'Using cached result from \'{cachefile}\'')
        result = pickle.load(cachehandle)
else:
    from load_data import load_data
    result = load_data(window=200, return_column_feature=True)

    if enable_cache:
        with open(cachefile, 'wb') as cachehandle:
            print(f'Saving result to cache \'{cachefile}\'')
            pickle.dump(result, cachehandle)

x_train, y_train, x_test, y_test, column_feature = result

print(column_feature)

print(x_train.shape, x_test.shape)
print(x_train)

print(y_train.shape, y_test.shape)
print(y_train)

window = x_train.shape[1]
feature = x_train.shape[2]
# output_class = y_train.shape[1]
# output_class = y_train.shape[1] + 1
output_class = 1

# DenseInputBias DenseAverage
from keras_layer import RelativePosition, relative_position, SimpleDense, DenseAverage, DenseBatchNormalization, DenseNotTainable

class DenseBlock(tensorflow.keras.layers.Layer):
    def __init__(self):
        super(DenseBlock, self).__init__()
        self.layer_1 = tensorflow.keras.layers.Dense(2)
        self.layer_2 = tensorflow.keras.layers.Activation('relu')
        self.layer_3 = tensorflow.keras.layers.Dense(256)
        self.layer_4 = tensorflow.keras.layers.BatchNormalization()
        self.layer_5 = tensorflow.keras.layers.Activation('relu')
        self.layer_6 = tensorflow.keras.layers.Dense(256)
        self.layer_7 = tensorflow.keras.layers.BatchNormalization()
        self.layer_8 = tensorflow.keras.layers.Activation('relu')
        self.layer_9 = tensorflow.keras.layers.Dense(2)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        x = self.layer_9(x)
        return x

class DenseBlockSkip(tensorflow.keras.layers.Layer):
    def __init__(self, n):
        super(DenseBlockSkip, self).__init__()
        self.layer_1 = tensorflow.keras.layers.BatchNormalization()
        self.layer_2 = tensorflow.keras.layers.Activation('relu')
        self.layer_3 = tensorflow.keras.layers.Dense(64)
        self.layer_4 = tensorflow.keras.layers.BatchNormalization()
        self.layer_5 = tensorflow.keras.layers.Activation('relu')
        self.layer_6 = tensorflow.keras.layers.Dense(n)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x += inputs
        return x

def define_model():
    '''
    class CustomModel(tensorflow.keras.Model):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.layer_1 = Conv1D(16, kernel_size=200)
            self.layer_2 = Flatten()
            self.layer_3 = RelativePosition(n=16)
            self.layer_4 = Flatten()
            self.layer_5 = Dense(64)
            self.layer_6 = Activation('relu')
            self.layer_7 = BatchNormalization()
            self.layer_8 = Dense(2)
            self.layer_9 = Activation('softmax')

        def call(self, inputs):
            x = self.layer_1(inputs)
            x = self.layer_2(x)
            x = self.layer_3(x)
            x = self.layer_4(x)
            x = self.layer_5(x)
            x = self.layer_6(x)
            x = self.layer_7(x)
            x = self.layer_8(x)
            x = self.layer_9(x)
            return x

    # inputs = Input(shape=(200, 1,))
    model = CustomModel()  # inputs=inputs, outputs=x
    '''

    inputs = tensorflow.keras.layers.Input(shape=(window, feature))

    '''
    x = tensorflow.keras.layers.Flatten()(inputs)
    b1 = DenseBlock()(x)
    b2 = DenseBlock()(x)
    b3 = DenseBlock()(x)
    b4 = DenseBlock()(x)
    b5 = DenseBlock()(x)
    b6 = DenseBlock()(x)
    b7 = DenseBlock()(x)
    b8 = DenseBlock()(x)
    b9 = DenseBlock()(x)
    b10 = DenseBlock()(x)
    b11 = DenseBlock()(x)
    x = tensorflow.keras.layers.Concatenate()([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11])
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(2)(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(256)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(256)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(2)(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(output_class)(x)
    x = tensorflow.keras.layers.Activation('softmax')(x)
    '''

    '''
    x = tensorflow.keras.layers.Flatten()(inputs)
    x = tensorflow.keras.layers.Dense(256)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(256)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(2)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(256)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(256)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(2)(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(256)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(256)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(2)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(256)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(256)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(output_class)(x)
    x = tensorflow.keras.layers.Activation('softmax')(x)
    '''

    '''
    x = tensorflow.keras.layers.Flatten()(inputs)
    x = tensorflow.keras.layers.Dense(256)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(256)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(2)(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(256)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(256)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(2)(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(256)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(256)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(output_class)(x)
    x = tensorflow.keras.layers.Activation('softmax')(x)
    '''

    x = tensorflow.keras.layers.Flatten()(inputs)
    x = tensorflow.keras.layers.Dense(output_class * 4 ** 4)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(output_class * 4 ** 4)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(output_class * 4 ** 4)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(output_class * 4 ** 3)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(output_class * 4 ** 2)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(output_class * 4)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(output_class)(x)
    # x = tensorflow.keras.layers.Activation('softmax')(x)

    '''
    x = tensorflow.keras.layers.Flatten()(inputs)
    x = tensorflow.keras.layers.Dense(128)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = DenseNotTainable(128)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(4)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(output_class)(x)
    x = tensorflow.keras.layers.Activation('softmax')(x)
    '''

    '''
    x = tensorflow.keras.layers.GRU(64)(inputs)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(16)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(output_class)(x)
    '''

    model = tensorflow.keras.models.Model(inputs=inputs, outputs=x)

    '''
    x = tensorflow.keras.layers.Flatten()(inputs)
    x = tensorflow.keras.layers.Dense(2)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(128)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(16)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(output_class)(x)
    x = tensorflow.keras.layers.Activation('softmax')(x)
    model = tensorflow.keras.models.Model(inputs=inputs, outputs=x)
    '''

    '''
    x = tensorflow.keras.layers.Conv1D(32, kernel_size=3)(inputs)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tensorflow.keras.layers.Conv1D(64, kernel_size=3)(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tensorflow.keras.layers.Conv1D(128, kernel_size=3)(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tensorflow.keras.layers.Flatten()(x)
    x = tensorflow.keras.layers.Dense(16)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Dense(output_class)(x)
    x = tensorflow.keras.layers.Activation('softmax')(x)
    model = tensorflow.keras.models.Model(inputs=inputs, outputs=x)
    '''

    '''
    x = Flatten()(inputs)
    x = SimpleDense(1250)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SimpleDense(1250)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SimpleDense(2)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    '''

    '''
    x = Flatten()(inputs)
    x = DenseAverage(1250)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DenseAverage(1250)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SimpleDense(2)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    '''

    '''
    x = Flatten()(inputs)
    x = DenseBatchNormalization(2048)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DenseBatchNormalization(64)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SimpleDense(2)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    '''

    '''
    x = Flatten()(inputs)
    x = Dense(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = DenseBlockSkip(1000)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(2)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    '''

    return model

with scope():
    # data_train = strategy.experimental_distribute_dataset(data_train)
    # data_test = strategy.experimental_distribute_dataset(data_test)

    try:
        model = tensorflow.keras.models.load_model('./model')
    except OSError as error:
        print(f'Model has not found: {error}')
        model = define_model()

    # steps_per_epoch = len(x_test) // batch_size
    # log.info(f'steps_per_epoch: {steps_per_epoch}')

    # learning_rate_schedule = tensorflow.keras.optimizers.schedules.InverseTimeDecay(
        # 0.001, decay_steps=steps_per_epoch * 1000, decay_rate=1, staircase=False
    # )

    # learning_rate=1e-6 learning_rate=learning_rate_schedule
    # accuracy accuracy60 ratio2 reverse_direction reverse_direction60
    # loss_custom steps_per_execution=len(x_test) // 200
    model.compile(optimizer=tensorflow.optimizers.SGD(), loss='mean_absolute_error', metrics=[error_absolute_maximum,
                  ratio_entry, ratio_exit_profit, ratio_hold, ratio_exit_loss, percent, percent_loss])
    model.summary()

    early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=10, mode='min', #  baseline=0.01,
                                                              start_from_epoch=20)
    while True:
        try:
            # data_train batch_size data_test
            history = model.fit(x_train, y_train, batch_size=200, epochs=300, validation_data=(x_test, y_test),
                                callbacks=[early_stopping])

        except KeyboardInterrupt:
            print('\nPaused: KeyboardInterrupt')
            # model.save('./model')
            break

sys.exit()
