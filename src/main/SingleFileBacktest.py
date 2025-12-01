import pandas as pd
import numpy as np
import sys
import os

from src.helper.BacktestHelper import BacktestHelper

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.6f}'.format)

# import data from excel
testDataPath = r"C:\Users\User\Desktop\backtesting\interval_1h-20250830T115325Z-1\interval_24h\train data\,derivatives,futures_volume_buy_sum-BTC-24h-bitget--native---#train.csv"
df = pd.read_csv(testDataPath)

window_list = np.round(np.arange(24, 192, 24), 1)
threshold_list = np.round(np.arange(0, 2, 0.25), 2)
ANNUALIZATION_FACTOR = 365 * 24 # 1h data
backtestHelper = BacktestHelper(df, 'zScore', ANNUALIZATION_FACTOR)
backtestHelper.generateHeatMap(window_list, threshold_list, 0)
# backtestHelper.singlePointBacktest(144, 0.25)
# backtestHelper.showSecondaryAxis()