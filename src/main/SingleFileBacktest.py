import pandas as pd
import numpy as np

from src.helper.BacktestHelper import BacktestHelper

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.6f}'.format)

# import data from excel
testDataPath = r"C:\Users\User\Desktop\backtesting\interval_1h-20250830T115325Z-1\interval_24h\train data\,derivatives,futures_volume_buy_sum-BTC-24h-bitget--native---#train.csv"
df = pd.read_csv(testDataPath)

window_list = np.round(np.arange(10, 100, 5), 1)
threshold_list = np.round(np.arange(0, 2, 0.25), 2)
backtestHelper = BacktestHelper(df, 'zScore')
# backtestHelper.generateHeatMap(window_list, threshold_list, 0)
backtestHelper.singlePointBacktest(35, 1.25)
# backtestHelper.showSecondaryAxis()