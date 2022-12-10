from pandas import DataFrame
from freqtrade.configuration import Configuration
from freqtrade.data.dataprovider import DataProvider
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.resolvers import StrategyResolver
import generate_dataset

# def column_feature(dataframe: DataFrame) -> list[str]:
def column_feature(dataframe: DataFrame) -> list:
    column_names = dataframe.columns
    feature = [c for c in column_names if c[0] == '%']
    return feature

# def column_label(dataframe: DataFrame) -> list[str]:
def column_label(dataframe: DataFrame) -> list:
    column_names = dataframe.columns
    label = [c for c in column_names if c[0] == '&']
    return label

def load_data(return_column_feature: bool = False):  # timerange: str
    config = Configuration.from_files(['./config.json'])
    config['timerange'] = '20210601-20220101'
    pair = 'ETH/USDT'

    strategy = StrategyResolver.load_strategy(config=config)
    strategy.freqai_info = config['freqai']
    strategy.dp = DataProvider(config=config, exchange=None, pairlists=None, rpc=None)

    timeframe = strategy.timeframe
    dataframe = strategy.dp.get_pair_dataframe(pair, timeframe)

    dk = FreqaiDataKitchen(config=config, live=False, pair=pair)
    dataframe = dk.use_strategy_to_populate_indicators(
        strategy=strategy, prediction_dataframe=dataframe, pair=pair
    )

    close = dataframe['close']
    volume = dataframe['volume']
    dataframe_feature = dataframe[column_feature(dataframe)]

    x_train, y_train, x_test, y_test = (
        generate_dataset.generate_dataset(dataframe_feature.to_numpy(dtype='float32'), close.to_numpy(dtype='float32'),
        # generate_dataset.generate_dataset(dataframe_feature.to_numpy(dtype='float'), close.to_numpy(dtype='float'),
                                          (volume > 0).to_numpy(dtype='bool'), window=1,
                                          threshold=0.01, batch_size=200, split_ratio=0.8, train_include_test=False,
                                          enable_window_nomalization=False)
    )

    if len(x_train) == 0 or len(x_test) == 0:
        raise Exception

    if not return_column_feature:
        return (x_train, y_train, x_test, y_test)
    else:
        return (x_train, y_train, x_test, y_test, column_feature(dataframe))
