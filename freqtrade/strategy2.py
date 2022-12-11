import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from functools import reduce
from typing import Literal, Optional, Union

import numpy
import pandas as pd
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
from technical import qtpylib

import indicator

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

def rci(close: numpy.ndarray,
        timeperiod: int = 9) -> numpy.ndarray:
    rank_target = [numpy.roll(close, i, axis=-1) for i in range(timeperiod)]
    rank_target = numpy.vstack(rank_target)[:, timeperiod - 1:]
    price_rank = numpy.argsort(numpy.argsort(rank_target[::-1], axis=0), axis=0) + 1
    time_rank = numpy.arange(1, timeperiod + 1).reshape(timeperiod, -1)
    aa = numpy.sum((time_rank - price_rank)**2, axis=0, dtype=float) * 6
    bb = float(timeperiod * (timeperiod**2 - 1))
    cc = numpy.divide(aa, bb, out=numpy.zeros_like(aa), where=bb != 0)
    rci = (1 - cc) * 100
    rci = numpy.concatenate([numpy.full(timeperiod - 1, numpy.nan), rci], axis=0)
    return rci

class Strategy2(IStrategy):
    def __init__(self, config: dict) -> None:
        super().__init__(config=config)
        self.path_runtime = './runtime.json'
        self.indent = 4

    def runtime_write(self) -> None:
        with open(self.path_runtime, mode='w') as file:
            json.dump(self.runtime, file, indent=self.indent, sort_keys=True)

    def runtime_load(self) -> None:
        with open(self.path_runtime, mode='r') as file:
            self.runtime = json.load(file)

    def runtime_pair_initial(self) -> dict:
        return {
            'previous': {
                'direction': None,
                'profit': None,
            },
            'stoploss': None,
            'takeprofit': None,
        }

    def runtime_pair_reset(self, pair: str) -> None:
        if pair not in self.runtime:
            raise Exception

        self.runtime[pair] = self.runtime_pair_initial()

    def runtime_update(self, list_pair: list) -> None:  # list[str]
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

        if self.dp.runmode.value != 'plot':
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
    stoploss: float = -1.00  # -100%
    # stoploss: float = -0.04  # -4%
    # use_custom_stoploss: bool = True

    plot_config = {
        "main_plot": {},
        "subplots": {
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    startup_candle_count: int = 2000

    def populate_any_indicators(self, pair: str, dataframe: DataFrame, timeframe: str, informative: DataFrame = None,
                                set_generalized_indicators: bool = False) -> DataFrame:
        coin = pair.split('/')[0]

        if informative is None:
            informative = self.dp.get_pair_dataframe(pair, timeframe)

        for t in self.freqai_info['feature_parameters']['indicator_periods_candles']:
            t = int(t)
            # informative[f'%{coin}-MFI-{t}'] = ta.MFI(informative, timeperiod=t) / 100
            # informative[f'%{coin}-ADX-{t}'] = ta.ADX(informative, timeperiod=t) / 100

            # bollinger = qtpylib.bollinger_bands(
                # qtpylib.typical_price(informative), window=t, stds=2.2
            # )
            # informative[f'{coin}-bb_lowerband-{t}'] = bollinger['lower']
            # informative[f'{coin}-bb_middleband-{t}'] = bollinger['mid']
            # informative[f'{coin}-bb_upperband-{t}'] = bollinger['upper']
            # informative[f'%{coin}-bb_width-{t}'] = (
                # informative[f'{coin}-bb_upperband-{t}']
                # - informative[f'{coin}-bb_lowerband-{t}']
            # ) / informative[f'{coin}-bb_middleband-{t}']
            # informative[f'%{coin}-close-bb_lower-{t}'] = (
                # informative['close'] / informative[f'{coin}-bb_lowerband-{t}']
            # )

            # informative[f'%{coin}-ROC-{t}'] = ta.ROC(informative, timeperiod=t)
            # informative[f'%{coin}-relative_volume-{t}'] = (
                # informative['volume'] / informative['volume'].rolling(t).mean()
            # )

            informative[f'{coin}-heikin_ashi-close'] = (
                (informative['open'] + informative['high'] + informative['low'] + informative['close']) / 4
            )

            # informative[f'%{coin}-RSI-{t}'] = ta.RSI(informative[f'{coin}-heikin_ashi-close'], timeperiod=t) / 100
            # informative[f'%{coin}-RCI-{t}'] = rci(informative[f'{coin}-heikin_ashi-close'].to_numpy(), timeperiod=t) / 100

            # informative[f'{coin}-RSI-{t}'] = ta.RSI(informative[f'{coin}-heikin_ashi-close'], timeperiod=t) / 100
            # informative[f'%{coin}-EMA(RSI)-{t}'] = ta.EMA(informative[f'{coin}-RSI-{t}'], timeperiod=t)
            # informative[f'%{coin}-WMA(RSI)-{t}'] = ta.WMA(informative[f'{coin}-RSI-{t}'], timeperiod=t)
            # informative[f'%{coin}-HMA(RSI)-{t}'] = qtpylib.hma(informative[f'{coin}-RSI-{t}'], window=t)
            # informative[f'%{coin}-Regression_1(RSI)-{t}'] = (
                # indicator.regression_1(informative[f'{coin}-RSI-{t}'].to_numpy(), window=t)
            # )
            # informative[f'%{coin}-moving_average_simple(RSI)-{t}'] = (
                # indicator.moving_average_simple(informative[f'{coin}-RSI-{t}'].to_numpy(), window=t)
            # )

            informative[f'{coin}-Group1-moving_average_simple-{t}'] = (
                indicator.moving_average_simple(informative[f'{coin}-heikin_ashi-close'].to_numpy(), window=t)
            )
            informative[f'{coin}-Group1-Regression_1-{t}'] = (
                indicator.regression_1(informative[f'{coin}-heikin_ashi-close'].to_numpy(), window=t)
            )
            informative[f'{coin}-Group1-EMA-{t}'] = ta.EMA(informative[f'{coin}-heikin_ashi-close'], timeperiod=t)
            informative[f'{coin}-Group1-WMA-{t}'] = ta.WMA(informative[f'{coin}-heikin_ashi-close'], timeperiod=t)
            informative[f'{coin}-Group1-HMA-{t}'] = qtpylib.hma(informative[f'{coin}-heikin_ashi-close'], window=t)

            # bollinger = qtpylib.bollinger_bands(informative[f'{coin}-heikin_ashi-close'], window=t, stds=2.2)
            # informative[f'{coin}-Group1-BB_lower-{t}'] = bollinger['lower']
            # informative[f'{coin}-Group1-BB_upper-{t}'] = bollinger['upper']

            column_group1 = [column for column in informative if column.startswith(f'{coin}-Group1-')]
            column = column_group1
            for i in range(len(column)):
                for j in range(i + 1, len(column)):
                    # informative[f'%({column[i]} - {column[j]})'] = informative[column[i]] - informative[column[j]]
                    informative[f'%({column[i]} / {column[j]})'] = informative[column[i]] / informative[column[j]]

        indicators = [col for col in informative if col.startswith('%')]
        for n in range(self.freqai_info['feature_parameters']['include_shifted_candles'] + 1):
            if n == 0:
                continue
            informative_shift = informative[indicators].shift(n)
            informative_shift = informative_shift.add_suffix('_shift-' + str(n))
            informative = pd.concat((informative, informative_shift), axis=1)

        dataframe = merge_informative_pair(dataframe, informative, self.config['timeframe'], timeframe, ffill=True)
        skip_columns = [
            (s + '_' + timeframe) for s in ['date', 'open', 'high', 'low', 'close', 'volume']
        ]
        dataframe = dataframe.drop(columns=skip_columns)

        if set_generalized_indicators:
            dataframe['%day_of_week'] = (dataframe['date'].dt.dayofweek + 1) / 7
            dataframe['%hour_of_day'] = (dataframe['date'].dt.hour + 1) / 24
            # dataframe['%minute_of_hour'] = (dataframe['date'].dt.minute + 1) / 60
            dataframe['&prediction'] = 0.

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # dataframe = self.freqai.start(dataframe, metadata, self)

        '''
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
        '''
        a = numpy.arange(len(dataframe)) % 2
        dataframe.loc[a == 1, ['enter_long', 'enter_tag']] = (1, 'Long')
        dataframe.loc[a == 0, ['enter_short', 'enter_tag']] = (1, 'Short')

        # dataframe.loc[(a == 1) | (a == 0), ['enter_long', 'enter_tag']] = (1, 'Long')
        # dataframe.loc[(a == 1) | (a == 0), ['enter_short', 'enter_tag']] = (1, 'Short')

        close = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        maximum = close.rolling(window=2000).max()
        minimum = close.rolling(window=2000).min()
        dataframe['minmax200'] = (maximum - minimum) / close / 5
        # print(dataframe[['date', 'minmax200']].to_markdown())

        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str,
                            current_time: datetime, entry_tag: Optional[str], side: str, **kwargs) -> bool:

        previous_direction = self.runtime[pair]['previous']['direction']
        previous_profit = self.runtime[pair]['previous']['profit']

        if previous_direction != side:
            return True

        return False

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
        # threshold = candle_last['minmax200']
        threshold = 0.02

        # if candle_last['&prediction'] == 0 or candle_last['&prediction'] == 1:
        if True:
            self.runtime[pair]['stoploss'] = -threshold
            # self.runtime[pair]['takeprofit'] = 0.
            self.runtime[pair]['takeprofit'] = threshold

        log.debug(f'runtime[pair]:\n{json_dumps(self.runtime[pair])}')

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

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle_last = dataframe.iloc[-1].squeeze()
        # threshold = candle_last['minmax200']
        threshold = 0.02

        # takeprofit_candidate = current_profit
        takeprofit_candidate = threshold + current_profit
        stoploss_candidate = -threshold + current_profit

        if takeprofit_candidate > self.runtime[pair]['takeprofit']:
            self.runtime[pair]['takeprofit'] = takeprofit_candidate

        if stoploss_candidate > self.runtime[pair]['stoploss']:
            self.runtime[pair]['stoploss'] = stoploss_candidate

        reason = None

        # if current_profit > threshold and current_profit < self.runtime[pair]['takeprofit'] / 4 * 3:
            # reason = f'takeprofit_{self.runtime[pair]["takeprofit"]:0.2f}'
        # if current_profit < 0 and current_profit < self.runtime[pair]['stoploss']:
            # reason = f'stoploss_{self.runtime[pair]["stoploss"]:0.2f}'

        if current_profit > self.runtime[pair]['takeprofit']:
            reason = f'takeprofit_{self.runtime[pair]["takeprofit"]:0.2f}'

        if current_profit < self.runtime[pair]['stoploss']:
            reason = f'stoploss_{self.runtime[pair]["stoploss"]:0.2f}'

        if reason is None:
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

        # self.runtime_pair_reset(pair)
        self.runtime[pair]['stoploss'] = None
        self.runtime[pair]['takeprofit'] = None
        self.runtime[pair]['previous']['direction'] = trade.trade_direction
        self.runtime[pair]['previous']['profit'] = (current_profit > 0)

        return reason

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float,
                        **kwargs) -> float:

        # dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # last_candle = dataframe.iloc[-1].squeeze()

        # Use parabolic sar as absolute stoploss price
        # stoploss_price = last_candle['sar']

        # Convert absolute price to percentage relative to current_rate
        # if stoploss_price < current_rate:
            # return (stoploss_price / current_rate) - 1

        # return maximum stoploss value, keeping current stoploss price unchanged
        # log.debug(f'runtime:\n{json_dumps(self.runtime)}')
        return 1.
        # return self.runtime[pair]['stoploss']

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
