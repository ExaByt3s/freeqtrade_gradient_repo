import logging
from typing import Any, Dict, Tuple

import pandas
from pandas import DataFrame
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import tensorflow as tf
# from freqtrade.freqai.base_models.BaseTensorFlowModel import BaseTensorFlowModel, WindowGenerator
from BaseTensorFlowModel import BaseTensorFlowModel, WindowGenerator
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Activation, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np

from tensorflow.keras.utils import to_categorical
import generate_dataset

logger = logging.getLogger(__name__)

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

class CNNPredictionModel2(BaseTensorFlowModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), fit().
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.CONV_WIDTH = 1

    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :params:
        :data_dictionary: the dictionary constructed by DataHandler to hold
        all the training and test data/labels.
        """
        # train_df = data_dictionary["train_features"]
        # train_labels = data_dictionary["train_labels"]
        # test_df = data_dictionary["test_features"]
        # test_labels = data_dictionary["test_labels"]
        # n_labels = len(train_labels.columns)

        n_features = len(dk.training_features_list)
        n_labels = len(dk.label_list)

        # print(list(data_dictionary))
        # print(train_labels.to_markdown())
        # print(len(train_labels))
        # print(list(train_df))
        # print(test_labels.to_markdown())
        # print(len(test_labels))

        if n_labels > 1:
            raise OperationalException(
                "Neural Net not yet configured for multi-targets. Please "
                " reduce number of targets to 1 in strategy."
            )

        # n_features = len(data_dictionary["train_features"].columns)
        batch_size = self.freqai_info.get("batch_size", 64)
        input_dims = [self.CONV_WIDTH, n_features]

        # test_labels_mask = (test_labels != -1)

        # w1 = WindowGenerator(
            # input_width=self.CONV_WIDTH,
            # label_width=1,
            # shift=0,
            # train_df=train_df,
            # val_df=test_df,
            # train_labels=train_labels,
            # val_labels=test_labels,
            # batch_size=batch_size,
        # )

        # dataframe = pandas.concat([train_df, test_df])
        dataframe = data_dictionary['unfiltered_df']
        dataframe = dataframe[dk.training_features_list]
        # print(list(dataframe))
        # print(dataframe['%-close'].to_markdown())
        # print(dataframe.to_markdown())
        x_train, y_train, x_test, y_test = (
            generate_dataset.generate_dataset(dataframe.to_numpy(dtype='float32'),
                                              dataframe['%-close'].to_numpy(dtype='float32'),
                                              (dataframe['%-volume'] > 0).to_numpy(dtype='bool'), window=self.CONV_WIDTH,
                                              threshold=0.02, batch_size=batch_size, split_ratio=0.8, train_include_test=True)
        )
        if len(x_train) == 0 or len(x_test) == 0:
            raise Exception

        # print(len(x_train))
        # print(len(x_test))
        # print(len(y_train))
        # print(len(y_test))
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        # print(len(x_train))
        dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        dataset_train = dataset_train.batch(batch_size, drop_remainder=True)
        dataset_test = dataset_test.batch(batch_size, drop_remainder=True)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=3, mode='min', min_delta=0.0001
            # monitor='val_accuracy', patience=3, mode='min', min_delta=0.0001
        )

        model = self.get_init_model(dk.pair)

        if model is None:
            logger.info('Create new model')

            model = self.create_model(input_dims, n_labels)

            # steps_per_epoch = np.ceil(len(test_df) / batch_size)
            steps_per_epoch = len(x_test) // batch_size
            print(f'steps_per_epoch: {steps_per_epoch}')

            lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                0.001, decay_steps=steps_per_epoch * 1000, decay_rate=1, staircase=False
            )

            model.compile(
                optimizer=tf.optimizers.Adam(lr_schedule),
                # loss=tf.losses.MeanSquaredError(),
                loss='categorical_crossentropy',
                # metrics=[tf.metrics.MeanAbsoluteError()],
                # metrics=[tf.metrics.MeanSquaredError()],
                metrics=['accuracy'],
            )
            model.summary()
            max_epochs = 100

        else:
            logger.info('Update old model')
            max_epochs = 20

        model.fit(
            # w1.train,
            dataset_train,
            epochs=max_epochs,
            shuffle=False,
            # validation_data=w1.val,
            validation_data=dataset_test,
            callbacks=[early_stopping],
            verbose=1,
        )

        return model

    def predict(
        self, unfiltered_dataframe: DataFrame, dk: FreqaiDataKitchen, first=True
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Filter the prediction features data and predict with it.
        :param: unfiltered_dataframe: Full dataframe for the current backtest period.
        :return:
        :prediction: np.array of prediction
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """

        dk.find_features(unfiltered_dataframe)
        filtered_dataframe, _ = dk.filter_features(
            unfiltered_dataframe, dk.training_features_list, training_filter=False
        )
        # filtered_dataframe = dk.normalize_data_from_metadata(filtered_dataframe)
        dk.data_dictionary["prediction_features"] = filtered_dataframe

        # optional additional data cleaning/analysis
        # self.data_cleaning_predict(dk)

        # dataframe = unfiltered_dataframe

        if first:
            full_df = dk.data_dictionary["prediction_features"]

            w1 = WindowGenerator(
                input_width=self.CONV_WIDTH,
                label_width=1,
                shift=0,
                test_df=full_df,
                batch_size=len(full_df),
            )
            prediction = self.model.predict(w1.inference)

        else:
            data = dk.data_dictionary["prediction_features"]
            data = tf.expand_dims(data, axis=0)
            prediction = self.model(data, training=False)

        # print(prediction)
        # prediction = prediction[:, 0, 0]
        prediction_probability = prediction
        prediction = np.argmax(prediction_probability, axis=1)
        print(prediction)
        pred_df = DataFrame(prediction, columns=dk.label_list)
        # pred_df = dk.denormalize_labels_from_metadata(pred_df)
        do_predict = np.ones(len(pred_df))
        return (pred_df, do_predict)

    def create_model(self, input_dims, n_labels) -> Any:
        # input_layer = Input(shape=(input_dims[0], input_dims[1]))
        # Layer_1 = Conv1D(filters=32, kernel_size=(self.CONV_WIDTH,), activation="relu")(input_layer)
        # Layer_3 = Dense(units=32, activation="relu")(Layer_1)
        # output_layer = Dense(units=n_labels)(Layer_3)
        # return Model(inputs=input_layer, outputs=output_layer)

        input_layer = Input(shape=(input_dims[0], input_dims[1]))
        x = Flatten()(input_layer)  # inputs
        x = Dense(32)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(2)(x)
        x = Activation('softmax')(x)
        model = Model(inputs=input_layer, outputs=x)  # inputs

        return model
