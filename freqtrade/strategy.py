import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from functools import reduce
from typing import Literal, Optional, Union

# import numpy
import pandas as pd
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
from technical import qtpylib

# log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)

log = logging.getLogger(__name__)
log_level: Literal[0, 1, 2] = 2

if log_level == 2:
    log.setLevel(logging.DEBUG)
elif log_level == 1:
    log.setLevel(logging.INFO)
elif log_level == 0:
    log.setLevel(logging.ERROR)

# See https://stackoverflow.com/questions/46807204/python-logging-duplicated
# See https://stackoverflow.com/questions/14058453/making-python-loggers-output-all-messages-to-stdout-in-addition-to-log-file
if not log.handlers:
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s() %(message)s'))
    log.addHandler(sh)

log.propagate = False

def json_dumps(object_: dict) -> str:
    return json.dumps(object_, indent=4, default=list, sort_keys=True)

class Strategy(IStrategy):
    def __init__(self, config: dict) -> None:
        super().__init__(config=config)
        self.path_runtime = './runtime.json'
        self.indent = 4
        self.threshold = [0.04]

    def runtime_write(self) -> None:
        with open(self.path_runtime, mode='w') as file:
            json.dump(self.runtime, file, indent=self.indent, sort_keys=True)

    def runtime_load(self) -> None:
        with open(self.path_runtime, mode='r') as file:
            self.runtime = json.load(file)

    def runtime_pair_initial(self) -> dict:
        return {
            'stoploss': None,
            'takeprofit': None,
        }

    def runtime_pair_reset(self, pair: str) -> None:
        if pair not in self.runtime:
            raise Exception

        self.runtime[pair] = self.runtime_pair_initial()

    # def runtime_update(self, list_pair: list[str]) -> None:
    def runtime_update(self, list_pair: list) -> None:
        runtime_old = self.runtime
        self.runtime = {}

        for pair in runtime_old:
            if runtime_old[pair] != self.runtime_pair_initial():
                self.runtime[pair] = runtime_old[pair]

        for pair in list_pair:
            if pair not in self.runtime:
                self.runtime[pair] = self.runtime_pair_initial()

    def bot_start(self, **kwargs) -> None:
        time_begin = time.perf_counter()

        if not self.dp:
            raise Exception('DataProvider is required')

        if self.dp.runmode.value == 'hyperopt':
            raise Exception('Hyperopt is not supported')

        if self.dp.runmode.value in ['dry_run', 'live']:
            if os.path.isfile(self.path_runtime):
                self.runtime_load()
            else:
                self.runtime = {}

        elif self.dp.runmode.value == 'backtest':
            self.runtime = {}

        list_pair = self.dp.current_whitelist()
        self.runtime_update(list_pair)
        # log.debug(f'runtime:\n{json_dumps(self.runtime)}')

        time_end = time.perf_counter()
        log.info(
            f'runmode:{self.dp.runmode.value}'
            f' timeframe:{self.timeframe}'
            f' {time_end - time_begin:0.4f} (second)'
        )

    def bot_loop_start(self, **kwargs) -> None:
        time_begin = time.perf_counter()

        if self.dp.runmode.value in ['dry_run', 'live']:
            self.runtime_write()

        time_end = time.perf_counter()
        log.info(
            f'runmode:{self.dp.runmode.value}'
            f' timeframe:{self.timeframe}'
            f' {time_end - time_begin:0.4f} (second)'
        )

    # Disable ROI
    # minimal_roi: dict[str, int] = {
    minimal_roi: dict = {
        '0': 10000  # 10000 * 100%
    }

    # Disable stoploss
    # stoploss: float = -1.00  # -100%
    stoploss: float = -0.04  # -4%

    plot_config = {
        "main_plot": {},
        "subplots": {
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    # this is the maximum period fed to talib (timeframe independent)
    startup_candle_count: int = 40

    def populate_any_indicators(self, pair: str, df: DataFrame, tf: str, informative: DataFrame = None,
                                set_generalized_indicators: bool = False) -> DataFrame:
        coin = pair.split('/')[0]

        if informative is None:
            informative = self.dp.get_pair_dataframe(pair, tf)

        # first loop is automatically duplicating indicators for time periods
        for t in self.freqai_info["feature_parameters"]["indicator_periods_candles"]:

            t = int(t)
            informative[f"%-{coin}rsi-period_{t}"] = ta.RSI(informative, timeperiod=t)
            informative[f"%-{coin}mfi-period_{t}"] = ta.MFI(informative, timeperiod=t)
            informative[f"%-{coin}adx-period_{t}"] = ta.ADX(informative, timeperiod=t)
            informative[f"%-{coin}sma-period_{t}"] = ta.SMA(informative, timeperiod=t)
            informative[f"%-{coin}ema-period_{t}"] = ta.EMA(informative, timeperiod=t)

            bollinger = qtpylib.bollinger_bands(
                qtpylib.typical_price(informative), window=t, stds=2.2
            )
            informative[f"{coin}bb_lowerband-period_{t}"] = bollinger["lower"]
            informative[f"{coin}bb_middleband-period_{t}"] = bollinger["mid"]
            informative[f"{coin}bb_upperband-period_{t}"] = bollinger["upper"]

            informative[f"%-{coin}bb_width-period_{t}"] = (
                informative[f"{coin}bb_upperband-period_{t}"]
                - informative[f"{coin}bb_lowerband-period_{t}"]
            ) / informative[f"{coin}bb_middleband-period_{t}"]
            informative[f"%-{coin}close-bb_lower-period_{t}"] = (
                informative["close"] / informative[f"{coin}bb_lowerband-period_{t}"]
            )

            informative[f"%-{coin}roc-period_{t}"] = ta.ROC(informative, timeperiod=t)

            informative[f"%-{coin}relative_volume-period_{t}"] = (
                informative["volume"] / informative["volume"].rolling(t).mean()
            )

        indicators = [col for col in informative if col.startswith("%")]
        # This loop duplicates and shifts all indicators to add a sense of recency to data
        for n in range(self.freqai_info["feature_parameters"]["include_shifted_candles"] + 1):
            if n == 0:
                continue
            informative_shift = informative[indicators].shift(n)
            informative_shift = informative_shift.add_suffix("_shift-" + str(n))
            informative = pd.concat((informative, informative_shift), axis=1)

        df = merge_informative_pair(df, informative, self.config["timeframe"], tf, ffill=True)
        skip_columns = [
            (s + "_" + tf) for s in ["date", "open", "high", "low", "close", "volume"]
        ]
        df = df.drop(columns=skip_columns)

        # Add generalized indicators here (because in live, it will call this
        # function to populate indicators during training). Notice how we ensure not to
        # add them multiple times
        if set_generalized_indicators:
            df['%day_of_week'] = (df['date'].dt.dayofweek + 1) / 7
            df['%hour_of_day'] = (df['date'].dt.hour + 1) / 25

            dataframe = df
            dataframe['&prediction'] = 0

        # print(df)
        # print(list(df))
        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        # print(dataframe[['date', '']].to_markdown())
        enter_long_conditions = [
            dataframe['do_predict'] == 1,
            dataframe['&prediction'] == 1,
        ]

        if enter_long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ['enter_long', 'enter_tag']
            ] = (1, dataframe['&prediction'].add_prefix('Long-'))

        enter_short_conditions = [
            dataframe['do_predict'] == 1,
            dataframe['&prediction'] == 0,
        ]

        if enter_short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ['enter_short', 'enter_tag']
            ] = (1, dataframe['&prediction'].add_prefix('Short-'))

        return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, proposed_stake: float,
                            min_stake: Optional[float], max_stake: float, leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        if min_stake > max_stake:
            log.info(
                f'Entry signal has skipped: min_stake > max_stake: {min_stake:0.4f} > {max_stake:0.4f}'
            )
            return 0.

        initial_stake = self.stake_amount

        if initial_stake < min_stake:
            initial_stake = math.ceil(min_stake)

        if initial_stake > max_stake:
            log.info(
                f'Entry signal has skipped: initial_stake > max_stake: {initial_stake:0.4f} > {max_stake:0.4f}'
            )
            return 0.

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle_last = dataframe.iloc[-1].squeeze()

        if candle_last['&prediction'] == 0 or candle_last['&prediction'] == 1:
            # self.threshold[]
            self.runtime[pair] = {
                'stoploss': -0.04,
                'takeprofit': 0.04,
            }

        # log.debug(f'runtime:\n{json_dumps(self.runtime)}')

        log.info(
            f'current_time:{current_time}'
            f' pair:{pair}'
            f' (initial_stake/stake_amount):({initial_stake}/{self.stake_amount})'
            f' side:{side}'
            f' entry_tag:{entry_tag}'
            f' min_stake:{min_stake:0.4f}'
            f' max_stake:{max_stake:0.4f}'
            f' current_rate:{current_rate:0.4f}'
        )

        return initial_stake

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float,
                    **kwargs) -> Optional[Union[str, bool]]:

        # if current_profit > 0.04:
            # print(current_profit)
            
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle_last = dataframe.iloc[-1].squeeze()

        if ((trade.trade_direction == 'long' and candle_last['enter_long'] == 1)
                or (trade.trade_direction == 'short' and candle_last['enter_short'] == 1)):

            # takeprofit_candidate = self.runtime[pair]['takeprofit'] + current_profit
            # stoploss_candidate = self.runtime[pair]['stoploss'] + current_profit
            takeprofit_candidate = 0.04 + current_profit
            stoploss_candidate = -0.04 + current_profit

            if takeprofit_candidate > self.runtime[pair]['takeprofit']:
                self.runtime[pair]['takeprofit'] = takeprofit_candidate

            if stoploss_candidate > self.runtime[pair]['stoploss']:
                self.runtime[pair]['stoploss'] = stoploss_candidate

        if current_profit > self.runtime[pair]['takeprofit']:
            reason = f'takeprofit_{self.runtime[pair]["takeprofit"]:0.2f}'
        elif current_profit < self.runtime[pair]['stoploss']:
            reason = f'stoploss_{self.runtime[pair]["stoploss"]:0.2f}'
        else:
            return False

        # color_ansi: dict[str, str] = {
        color_ansi: dict = {
            'black'  : '\x1b[0;30m',
            'blue'   : '\x1b[0;34m',
            'cyan'   : '\x1b[0;36m',
            'green'  : '\x1b[0;32m',
            'magenda': '\x1b[0;35m',
            'red'    : '\x1b[0;31m',
            'yellow' : '\x1b[0;33m',
            'reset'  : '\x1b[0m'   ,
        }

        color_begin = ''
        color_end = color_ansi['reset']

        if current_profit > 0:
            color_begin = color_ansi['green']
        elif current_profit < 0:
            color_begin = color_ansi['red']

        # log.debug(f'runtime:\n{json_dumps(self.runtime)}')

        log.info(
            f'{color_begin}'
            f'current_time:{current_time}'
            f' pair:{pair}'
            f' reason:{reason}'
            f' current_profit:{current_profit}'
            f' open_rate:{trade.open_rate:0.4f}'
            f' current_rate:{current_rate:0.4f}'
            f' timedelta:{current_time - trade.open_date_utc}'
            f'{color_end}'
        )

        self.runtime_pair_reset(pair)
        return reason

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
