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
from ta.volatility import KeltnerChannel


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


# def add_trend(df, column):
#     df["trend"] = list()
#
#     for val in df[column]:
#
#
# def get_inflexion_points(df, column):
#     inflexion_indices = list()
#
#     # Compute all inflexion points using first derivative
#     for val in df[inflexion_indices]:
#
#
def compute_profitability(df, buy_indices):

    take_profit_diff = 0.2
    stop_loss_diff = 0.1

    profit = 0
    profit_evolution = []

    stop_loss_count = 0
    take_profit_count = 0;

    for index in buy_indices:
        start_price = df["Close"][index]
        step = index + 1
        while True:
            step_price = df["Close"][step]
            if step_price > start_price + take_profit_diff:
                profit += step_price - start_price
                profit_evolution.append(profit)
                stop_loss_count += 1
                break
            if step_price < start_price - stop_loss_diff:
                profit += step_price - start_price
                profit_evolution.append(profit)
                take_profit_count += 1
                break
            step += 1

    print("Profit: {}".format(profit))
    print("Take Profit: {}".format(take_profit_count))
    print("Stop Loss: {}".format(stop_loss_count))
    plt.plot(profit_evolution)
    plt.show()

    print("Done")

def get_inflexion_points(df):
    inflexion_points = []

    # for val in df["EMA"]:


def run():
    # Load datas
    # df = pd.read_csv('C:/Projects/QuantRat/TaDemo/ta/ta/tests/data/datas.csv', sep=',', nrows=10000)
    # df = pd.read_csv('C:/Projects/QuantRat/data/test_ta.csv', sep=',')
    # df = dropna(df)
    # df = tick_to_candle(df)

    df = pd.read_csv("C:/Projects/QuantRat/data/IVE_bidask1min.txt", nrows=500)
    # df["Close"].plot()

    # Add Keltner channel
    k_channel = KeltnerChannel(df["High"], df["Low"], df["Close"], 10, 10, ov=False)
    df["ChannelHigh"] = k_channel.keltner_channel_hband()
    df["ChannelLow"] = k_channel.keltner_channel_lband()
    df["ChannelMid"] = k_channel.keltner_channel_mband()

    # Add EMA
    # ema_slow = EMAIndicator(df["Close"], 20)
    # ema_fast = EMAIndicator(df["Close"], 10)
    # df["EMA_slow"] = ema_slow.ema_indicator()
    # df["EMA_fast"] = ema_fast.ema_indicator()

    # Plot
    plt.plot(df["Close"], "-b", label="Close")
    plt.plot(df["ChannelHigh"], "--c", label="K high band")
    plt.plot(df["ChannelLow"], "--y", label="K low band")
    plt.plot(df["ChannelMid"], "--m", label="K mid band")

    # plt.plot(df["EMA_fast"], "-r", label="EMA fast")
    # plt.plot(df["EMA_slow"], "-g", label="EMA slow")
    # plt.plot(df["EMA10"], "--m", label="EMA10")
    plt.legend(loc="upper right")
    plt.show()


    # compute_profitability(df, buy_signals[0])

    print("ready")