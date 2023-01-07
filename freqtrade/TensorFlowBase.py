import logging
from time import time
from typing import Any

import pandas
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel

logger = logging.getLogger(__name__)


class TensorFlowBase(IFreqaiModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config=config)
        self.keras = True

        if self.ft_params.get("DI_threshold", 0):
            self.ft_params["DI_threshold"] = 0
            logger.warning("DI threshold is not configured for Keras models yet. Deactivating.")

    def train(
        self, unfiltered_df: pandas.DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        # print(unfiltered_df)
        # print(list(unfiltered_df))
        start_time = time()

        # filter the features requested by user in the configuration file and elegantly handle NaNs
        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )

        # split data into train/test data.
        data_dictionary = dk.make_train_test_datasets(features_filtered, labels_filtered)
        if not self.freqai_info.get("fit_live_predictions", 0) or not self.live:
            dk.fit_labels()
        # normalize all data based on train_dataset only
        data_dictionary = dk.normalize_data(data_dictionary)

        # optional additional data cleaning/analysis
        self.data_cleaning_train(dk)

        logger.info(
            f"Training model on {len(dk.data_dictionary['train_features'].columns)} features"
        )
        logger.info(f"Training model on {len(data_dictionary['train_features'])} data points")

        data_dictionary['unfiltered_df'] = unfiltered_df
        model = self.fit(data_dictionary, dk)

        end_time = time()

        logger.info(f"{pair} ({end_time - start_time:.2f} secs)")

        return model
