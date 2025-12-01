import pandas as pd
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# src/main -> src -> crypto (root)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from src.helper.BacktestHelper import BacktestHelper

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.6f}'.format)

# import data from excel
testDataPath = os.path.join(project_root, 'src', 'testdata', 'interval_1h', 'train data', ',addresses,min_100_count-BTC-1h--aggregated-native---#train.csv')
df = pd.read_csv(testDataPath)

window_list = np.round(np.arange(24, 192, 24), 1)
threshold_list = np.round(np.arange(0, 2, 0.25), 2)
ANNUALIZATION_FACTOR = 365 * 24 # 1h data
backtestHelper = BacktestHelper(df, 'zScore', ANNUALIZATION_FACTOR)
backtestHelper.generateHeatMap(window_list, threshold_list, 0)
# backtestHelper.singlePointBacktest(144, 0.25)
# backtestHelper.showSecondaryAxis()