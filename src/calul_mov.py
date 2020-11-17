'''
Copyright 2020 by Traista Viorel-Rafael
All rights reserved.
This file is part of the Traista Viorel-Rafael - QuantRat project
https://github.com/TraistaRafael/QuantRat
Please see the LICENSE.txt
'''

import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.momentum import StochasticOscillator
from ta.momentum import StochRSIIndicator
from ta.trend import EMAIndicator

import plotly.graph_objects as go
import matplotlib.pyplot as plt

def tick_to_candle(df_tick, candle_seconds = 60):
    df_candle = pd.DataFrame({"Timestamp": list(), "Open": list(), "High": list(), "Low": list(), "Close": list()})

    last_minute_timestamp = float(df_tick["timestamp"][0])

    candle_open = float(df_tick["bidPrice"][0])
    candle_high = candle_open
    candle_low = candle_open

    for index, row in df_tick.iterrows():
        current_timestamp = float(row["timestamp"])
        bid_price = float(row["bidPrice"])

        if bid_price > candle_high:
            candle_high = bid_price

        if bid_price < candle_low:
            candle_low = bid_price

        if current_timestamp - last_minute_timestamp >= candle_seconds * 1000:
            # Add one more minute
            # print ("{}, {}, {}, {}".format(candle_open, candle_high, candle_low, bid_price))

            df_candle.loc[len(df_candle.index)] = [current_timestamp, candle_open, candle_high, candle_low, bid_price]
            last_minute_timestamp = current_timestamp

            candle_open = bid_price
            candle_high = candle_open
            candle_low = candle_open



    return df_candle


def tick_to_minute(df_tick):
    df_minute = pd.DataFrame({"Timestamp": list(), "Close": list()})

    last_minute_timestamp = float(df_tick["Timestamp"][0])

    for index, row in df_tick.iterrows():
        current_timestamp = float(row["Timestamp"])
        if current_timestamp - last_minute_timestamp >= 60 * 1000:
            # Add one more minute
            close_price = float(row["Close"])
            df_minute.loc[len(df_minute.index)] = [current_timestamp, close_price]
            last_minute_timestamp = current_timestamp

    return df_minute

def test_bollinger(df):
    # Initialize Bollinger Bands Indicator
    indicator_bb = BollingerBands(close=df["Close"], n=20, ndev=2)

    # Add Bollinger Bands features
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()

    # Add Bollinger Band high indicator
    df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()

    # Add Bollinger Band low indicator
    df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

    # Add Width Size Bollinger Bands
    df['bb_bbw'] = indicator_bb.bollinger_wband()

    # Add Percentage Bollinger Bands
    df['bb_bbp'] = indicator_bb.bollinger_pband()

    df[["Close", "bb_bbm", "bb_bbh", "bb_bbl"]].plot()


def plot(df, buy_signals = None):
    fig, axs = plt.subplots(2)

    axs[0].plot(df["Close"], "-b", label="Close")
    axs[0].plot(df["EMA50"], "--c", label="EMA50")
    axs[0].plot(df["EMA100"], "--m", label="EMA1000")

    # Plot buy signals
    if buy_signals is not None:
        axs[0].plot(buy_signals[0], buy_signals[1], "*r", label="BUY SIGNALS")

    axs[0].legend(loc="upper right")


    axs[1].plot(df["Stochastic"], "--b", label="SO 5-3")
    # axs[1].plot(df["StochasticRSI"], "--r", label="SORSI 5-3-3")
    axs[1].plot([0, len(df.index)], [20, 20], "--y")
    axs[1].plot([0, len(df.index)], [80, 80], "--y")
    axs[1].legend(loc="upper right")
    plt.show()

    # fig = go.Figure(data=[go.Candlestick(x=df.index,
    #                                      open=df['Open'],
    #                                      high=df['High'],
    #                                      low=df['Low'],
    #                                      close=df['Close'])])
    # fig.show()

def add_indicators(df):
    # Add EMA 50
    ema_50 = EMAIndicator(df["Close"], 50)
    df["EMA50"] = ema_50.ema_indicator()

    # Add EMA 100
    ema_100 = EMAIndicator(df["Close"], 100)
    df["EMA100"] = ema_100.ema_indicator()

    # Add Stochastic RSI Oscillator
    # stochasticRSI = StochRSIIndicator(df["Close"], 5, 3, 3)
    # df["StochasticRSI"] = stochasticRSI.stochrsi()

    # Add Stochastic Oscillator
    stochastic = StochasticOscillator(df["High"], df["Low"], df["Close"], 5, 3)
    df["Stochastic"] = stochastic.stoch()


def analyze(df):
    buy_index_list = list()
    buy_index_price = list()

    sell_index_list = list()
    sell_index_price = list()

    trend = "neutral"
    last_ema_delta = abs(df["EMA50"][0] - df["EMA100"][0])
    buy_strategy_stage = "search"  #trend_up, under80, pullback, under20, above20

    for index, row in df.iterrows():

        # 1 - check for trend changes
        if float(row["EMA50"]) > float(row["EMA100"]):
            if trend == "down" and buy_strategy_stage == "search":
                trend = "up"
                buy_strategy_stage = "trend_up"
                continue
        else:
            trend = "down"
            # reset progress
            buy_strategy_stage = "search"
            continue


        # 2 - Wait for Stochastic Osciallator drop under 80
        if buy_strategy_stage == "trend_up":
            if float(row["Stochastic"]) < 80:
                buy_strategy_stage = "under80"
                last_ema_delta =  abs(float(row["EMA50"]) - float(row["EMA100"]))
            continue

        # 3 - Check for EMA pullback
        if buy_strategy_stage == "under80":
            ema_delta = abs(float(row["EMA50"]) - float(row["EMA100"]))
            if ema_delta < last_ema_delta:
                #EMA50 is going back to 100
                buy_strategy_stage = "pullback"
            last_ema_delta = ema_delta
            continue

        # 4 - Check for Stochastic oscillator going below 20
        if buy_strategy_stage == "pullback":
            if float(row["Stochastic"]) < 20:
                buy_strategy_stage = "below20"
            continue

        # 5 - Check for Stochastic oscillator going above 20
        if buy_strategy_stage == "below20":
            if float(row["Stochastic"]) > 20:
                # BUY SIGNAL
                buy_index_list.append(index)
                buy_index_price.append(float(row["Close"]))

                #reset strategy
                buy_strategy_stage = "search"
            continue

    return buy_index_list, buy_index_price


def compute_profitability(df, buy_signals):
    return None


def run():
    # Load datas
    # df = pd.read_csv('C:/Projects/QuantRat/TaDemo/ta/ta/tests/data/datas.csv', sep=',', nrows=10000)
    # df = pd.read_csv('C:/Projects/QuantRat/data/test_ta.csv', sep=',')
    # df = dropna(df)
    # df = tick_to_candle(df)

    df = pd.read_csv("C:/Projects/QuantRat/data/IVE_bidask1min.txt", nrows=2000)
    # df["Close"].plot()


    add_indicators(df)

    buy_signals = analyze(df)

    plot(df, buy_signals)






    print("ready")