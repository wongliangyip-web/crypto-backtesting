import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

class BacktestHelper:
    def __init__(self, rawDf, backtestType, annualizationFactor=365):
        self.rawDf = rawDf
        self.backtestType = backtestType
        self.annualizationFactor = annualizationFactor
        self.df = self.rawDf.copy()

        df = self.rawDf.copy()

        if 'value' in df.columns:
            df['value'] = df['value']
        elif 'v' in df.columns:
            df['value'] = df['v']
        else:
            raise KeyError("Neither 'value' nor 'v' column found in the input DataFrame.")

        if 'human_time' not in df.columns:
            df['human_time'] = pd.to_datetime(df['t'])

        df = df[['t', 'price', 'value', 'human_time']]

        self.df = df

    def __zScoreBacktest(self, window, threshold, isReversePos=False):
        df = self.df

        df['chg'] = df['price'].pct_change()

        df['vma'] = df['value'].rolling(window).mean()
        df['sd'] = df['value'].rolling(window).std()
        zScore = (df['value'] - df['vma']) / df['sd']

        # --- Vectorized signal generation ---
        posCondition = [-1, 1] if isReversePos else [1, -1]
        df['pos'] = np.select(
            [zScore > threshold, zScore < -threshold],
            posCondition,
            default=0
        )

        # --- Strategy PnL and metrics ---
        df['pos_t-1'] = df['pos'].shift(1)
        df['pnl'] = df['pos_t-1'] * df['chg']
        df['cumu'] = df['pnl'].cumsum()
        df['dd'] = df['cumu'].cummax() - df['cumu']
        df['trade_flag'] = df['pos'] != df['pos_t-1']

        # --- Buy and Hold baseline ---
        df['bnh_pnl'] = df['chg']
        df.loc[:window - 1, 'bnh_pnl'] = 0
        df['bnh_cumu'] = df['bnh_pnl'].cumsum()
        num_trades = df['trade_flag'].sum()

        # --- Performance statistics ---
        valid_pnl = df['pnl'].dropna()

        if valid_pnl.std() == 0 or len(valid_pnl) < 2:
            sharpe = np.nan
            preciseSharpe = np.nan
        else:
            annual_return = round(valid_pnl.mean() * self.annualizationFactor, 3)
            sharpe = round(valid_pnl.mean() / valid_pnl.std() * np.sqrt(self.annualizationFactor), 3)
            mdd = df['dd'].max()
            calmar = round(annual_return / mdd, 3) if mdd != 0 else np.nan

            averageReturn = valid_pnl.mean()
            sd = valid_pnl.std()
            preciseSharpe = round(averageReturn / sd * np.sqrt(self.annualizationFactor), 3)

            print(df)
            print(window, threshold, num_trades, 'annual_return', annual_return,
                  'sharpe', sharpe, 'calmar', calmar, 'preciseSharpe', preciseSharpe)

        return pd.Series([window, threshold, sharpe, num_trades], index=['window', 'threshold', 'sharpe', 'trades'])

    def __maDiffBacktest(self, window, threshold, isReversePos=False):
        df = self.df
        df['chg'] = df['price'].pct_change()

        df['vma'] = df['value'].rolling(window).mean()

        # --- Vectorized signal generation ---
        posCondition = [-1, 1] if isReversePos else [1, -1]
        ratio = df['value'] / df['vma'] - 1
        df['pos'] = np.select(
            [ratio > threshold, ratio < -threshold],
            posCondition,
            default=0
        )

        # --- Strategy PnL and metrics ---
        df['pos_t-1'] = df['pos'].shift(1)
        df['pnl'] = df['pos_t-1'] * df['chg']
        df['cumu'] = df['pnl'].cumsum()
        df['dd'] = df['cumu'].cummax() - df['cumu']
        df['trade_flag'] = df['pos'] != df['pos_t-1']

        # --- Buy and Hold baseline ---
        df['bnh_pnl'] = df['chg']
        df.loc[:window - 1, 'bnh_pnl'] = 0
        df['bnh_cumu'] = df['bnh_pnl'].cumsum()
        num_trades = df['trade_flag'].sum()

        # --- Performance statistics ---
        valid_pnl = df['pnl'].dropna()

        if valid_pnl.std() == 0 or len(valid_pnl) < 2:
            sharpe = np.nan
            preciseSharpe = np.nan
        else:
            annual_return = round(valid_pnl.mean() * self.annualizationFactor, 3)
            sharpe = round(valid_pnl.mean() / valid_pnl.std() * np.sqrt(self.annualizationFactor), 3)
            mdd = df['dd'].max()
            calmar = round(annual_return / mdd, 3) if mdd != 0 else np.nan

            averageReturn = valid_pnl.mean()
            sd = valid_pnl.std()
            preciseSharpe = round(averageReturn / sd * np.sqrt(self.annualizationFactor), 3)

            print(df)
            print(window, threshold, num_trades, 'annual_return', annual_return,
                  'sharpe', sharpe, 'calmar', calmar, 'preciseSharpe', preciseSharpe)

        return pd.Series([window, threshold, sharpe, num_trades], index=['window', 'threshold', 'sharpe', 'trades'])

    def __simpleThresholdBacktest(self, upperThreshold, lowerThreshold, isReversePos=False):
        df = self.df
        df['chg'] = df['price'].pct_change()

        # --- Vectorized signal generation ---
        posCondition = [-1, 1] if isReversePos else [1, -1]
        df['pos'] = np.select(
            [df['value'] > upperThreshold, df['value'] < lowerThreshold],
            posCondition,
            default=0
        )

        # --- Strategy PnL and metrics ---
        df['pos_t-1'] = df['pos'].shift(1)
        df['pnl'] = df['pos_t-1'] * df['chg']
        df['cumu'] = df['pnl'].cumsum()
        df['dd'] = df['cumu'].cummax() - df['cumu']
        df['trade_flag'] = df['pos'] != df['pos_t-1']

        # --- Buy and Hold baseline ---
        df['bnh_pnl'] = df['chg']
        df['bnh_cumu'] = df['bnh_pnl'].cumsum()
        num_trades = df['trade_flag'].sum()

        # --- Performance statistics ---
        valid_pnl = df['pnl'].dropna()

        if valid_pnl.std() == 0 or len(valid_pnl) < 2:
            sharpe = np.nan
            preciseSharpe = np.nan
        else:
            annual_return = round(valid_pnl.mean() * self.annualizationFactor, 3)
            sharpe = round(valid_pnl.mean() / valid_pnl.std() * np.sqrt(self.annualizationFactor), 3)
            mdd = df['dd'].max()
            calmar = round(annual_return / mdd, 3) if mdd != 0 else np.nan

            averageReturn = valid_pnl.mean()
            sd = valid_pnl.std()
            preciseSharpe = round(averageReturn / sd * np.sqrt(self.annualizationFactor), 3)

            print(df)
            print(upperThreshold, lowerThreshold, num_trades, 'annual_return', annual_return,
                  'sharpe', sharpe, 'calmar', calmar, 'preciseSharpe', preciseSharpe)

        return pd.Series([upperThreshold, lowerThreshold, sharpe, num_trades], index=['window', 'threshold', 'sharpe', 'trades'])

    def generateHeatMap(self, window_list, threshold_list, isReversePos=False):
        ## optimisation
        result_list = []

        if self.backtestType == 'maDiff':
            backtestFunction = self.__maDiffBacktest
        elif self.backtestType == 'zScore':
            backtestFunction = self.__zScoreBacktest
        elif self.backtestType == 'simpleThreshold':
            backtestFunction = self.__simpleThresholdBacktest

        for window in window_list:
            for threshold in threshold_list:
                result = backtestFunction(window, threshold, isReversePos)
                result_list.append(result)

        result_df = pd.DataFrame(result_list)
        result_df = result_df.sort_values(by='sharpe', ascending=False)
        print(result_df)

        data_table = result_df.pivot(index='window', columns='threshold', values='sharpe')
        sns.heatmap(data_table, annot=True, cmap='Greens')
        plt.show()

        trades_table = result_df.pivot(index='window', columns='threshold', values='trades')
        sns.heatmap(trades_table, annot=True, fmt=",", cmap="Blues")
        plt.title("Number of Trades Heatmap")
        plt.show()

    def showSecondaryAxis(self):
        fig, ax1 = plt.subplots()

        ax1.plot(self.df['human_time'], self.df['price'], label='price')
        ax1.set_xlabel('time')
        ax1.set_ylabel('price', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(self.df['human_time'], self.df['value'], color='tab:red', label='value')
        ax2.set_ylabel('value', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title('Price Vs Value')
        plt.show()

    def singlePointBacktest(self, window, threshold, isReversePos=False):
        if self.backtestType == 'maDiff':
            backtestFunction = self.__maDiffBacktest
        elif self.backtestType == 'zScore':
            backtestFunction = self.__zScoreBacktest
        elif self.backtestType == 'simpleThreshold':
            backtestFunction = self.__simpleThresholdBacktest

        backtestFunction(window, threshold, isReversePos)
        fig = px.line(self.df, x='human_time', y=['cumu', 'dd', 'bnh_cumu'])
        fig.show()

    def findBestSharpe(self, window_list, threshold_list, isReversePos=False):
        result_list = []

        if self.backtestType == 'maDiff':
            backtestFunction = self.__maDiffBacktest
        elif self.backtestType == 'zScore':
            backtestFunction = self.__zScoreBacktest
        elif self.backtestType == 'simpleThreshold':
            backtestFunction = self.__simpleThresholdBacktest

        for window in window_list:
            for threshold in threshold_list:
                result = backtestFunction(window, threshold, isReversePos)
                result_list.append(result)

        result_df = pd.DataFrame(result_list)
        return result_df['sharpe'].max()