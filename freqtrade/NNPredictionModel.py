import logging
from typing import Any, Dict, Tuple

import numpy
import pandas
import tensorflow
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from pandas import DataFrame

import generate_dataset
from BaseTensorFlowModel import BaseTensorFlowModel, WindowGenerator
from load_data import column_feature, column_label

log = logging.getLogger(__name__)

class NNPredictionModel(BaseTensorFlowModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.CONV_WIDTH = 200

    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen) -> Any:
        dataframe = data_dictionary['unfiltered_df']

        for i in dk.label_list:
            if dataframe[i].dtype != 'float64':
                raise OperationalException

        n_features = len(dk.training_features_list)
        n_labels = len(dk.label_list)

        batch_size = self.freqai_info.get('batch_size', 200)
        input_dims = [self.CONV_WIDTH, n_features]

        feature = dataframe[column_feature(dataframe)].to_numpy(dtype='float32')
        feature_mask = (dataframe['volume'] > 0).to_numpy(dtype='bool')

        label = dataframe[column_label(dataframe)].to_numpy(dtype='float32')
        label_mask = numpy.full(len(label), True, dtype='bool')

        x_train, y_train, x_test, y_test = (
            generate_dataset.generate_dataset(feature, feature_mask, label, label_mask, window=self.CONV_WIDTH,
                                              batch_size=batch_size, split_ratio=0.95, train_include_test=False,
                                              enable_window_nomalization=True)  # train_include_test=True
        )

        if len(x_train) == 0 or len(x_test) == 0:
            raise Exception

        dataset_train = tensorflow.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset_test = tensorflow.data.Dataset.from_tensor_slices((x_test, y_test))

        dataset_train = dataset_train.batch(batch_size, drop_remainder=True)
        dataset_test = dataset_test.batch(batch_size, drop_remainder=True)

        early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=3, mode='min', min_delta=0.0001)
        model = self.get_init_model(dk.pair)

        if model is None:
            log.info('Creating new model')

            model = self.create_model(input_dims, n_labels)

            steps_per_epoch = len(x_test) // batch_size
            log.info(f'steps_per_epoch: {steps_per_epoch}')

            learning_rate_schedule = tensorflow.keras.optimizers.schedules.InverseTimeDecay(
                0.001, decay_steps=steps_per_epoch * 1000, decay_rate=1, staircase=False
            )

            model.compile(
                optimizer=tensorflow.optimizers.Adam(learning_rate_schedule),
                loss='mean_absolute_error',
                metrics=[],
            )
            model.summary()
            max_epochs = 30

        else:
            log.info('Updating old model')
            max_epochs = 10

        model.fit(dataset_train, epochs=max_epochs, shuffle=False, validation_data=dataset_test, callbacks=[early_stopping])
        return model

    def predict(self, unfiltered_dataframe: DataFrame, dk: FreqaiDataKitchen, first=True) -> Tuple[DataFrame, DataFrame]:
        dataframe = unfiltered_dataframe
        feature = dataframe[column_feature(dataframe)]

        # print(feature.to_markdown())
        # print(feature.shape)

        if first:
            log.info('First')
            x, x_mask = generate_dataset.generate_dataset_predict(
                            x=feature.to_numpy(dtype='float32'),
                            x_mask=numpy.full(len(feature), True, dtype='bool'), window=self.CONV_WIDTH, batch_size=200,
                            enable_window_nomalization=True)

            print(x)
            prediction = self.model.predict(x)

        else:
            log.info('Not first')
            data = feature
            # data = tensorflow.expand_dims(data, axis=0)
            print(data.shape)
            prediction = self.model(data, training=False)

        print(prediction)
        print(prediction.shape)

        pred_df = DataFrame(prediction, columns=dk.label_list)
        do_predict = numpy.ones(len(pred_df))
        return (pred_df, do_predict)

    def create_model(self, input_dims, n_labels) -> Any:
        output_class = n_labels
        inputs = tensorflow.keras.layers.Input(shape=(input_dims[0], input_dims[1]))
        x = tensorflow.keras.layers.Flatten()(inputs)
        x = tensorflow.keras.layers.Dense(16)(x)
        x = tensorflow.keras.layers.BatchNormalization()(x)
        x = tensorflow.keras.layers.Activation('relu')(x)
        x = tensorflow.keras.layers.Dense(4)(x)
        x = tensorflow.keras.layers.BatchNormalization()(x)
        x = tensorflow.keras.layers.Activation('relu')(x)
        x = tensorflow.keras.layers.Dense(output_class)(x)
        model = tensorflow.keras.models.Model(inputs=inputs, outputs=x)
        return model
