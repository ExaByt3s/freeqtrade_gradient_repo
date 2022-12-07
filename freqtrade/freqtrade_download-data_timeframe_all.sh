#!/bin/sh
TIMERANGE='20000101-20221201'
# 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M
TIMEFRAME='5m'

freqtrade download-data --timeframes ${TIMEFRAME} --timerange "${TIMERANGE}"
#  --trading-mode futures -p BTC/USDT ETH/USDT
