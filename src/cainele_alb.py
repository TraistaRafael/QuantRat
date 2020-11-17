'''
Copyright 2020 by Traista Viorel-Rafael
All rights reserved.
This file is part of the Traista Viorel-Rafael - QuantRat project
https://github.com/TraistaRafael/QuantRat
Please see the LICENSE.txt
'''

import FinanceUtils

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier)
import matplotlib.pyplot as plt
import statistics as stat
from scipy.stats import linregress
from matplotlib.pyplot import figure
import time
import random
from os import path

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------------------------------------------------------------------------------------
# BEGIN INDICATORS
# ---------------------------------------------------------------------------------------------------

'''
Receive pandas timeseries
Returns pandas timeseries
'''
def moving_average_one_sec(tick_data, seconds):
    avg_data = pd.DataFrame({"second": list(), "timestamp":list(), "askPrice": list(), "bidPrice": list(), "askSize": list(), "bidSize": list()})

    sum_ask_price = 0
    sum_bid_price = 0
    sum_ask_size = 0
    sum_bid_size = 0

    current_millisecond = 0
    current_second = 0
    tick_count = 0
    index = 0

    last_tick = float(tick_data["timestamp"][0])

    for index, row in tick_data.iterrows():
        sum_ask_price += float(row["askPrice"])
        sum_bid_price += float(row["bidPrice"])
        sum_ask_size += float(row["askSize"])
        sum_bid_size += float(row["bidSize"])
        tick = float(row["timestamp"])
        tick_count += 1

        current_millisecond += tick - last_tick
        last_tick = tick

        if current_millisecond > seconds * 1000:
            sum_ask_price /= tick_count
            sum_bid_price /= tick_count
            sum_ask_size /= tick_count
            sum_bid_size /= tick_count

            step_seconds = int(current_millisecond / (seconds * 1000))

            for i in range(step_seconds):
                avg_data.loc[index] = [current_second, tick, sum_ask_price, sum_bid_price, sum_ask_size, sum_bid_size]
                current_second += seconds
                index += 1
                tick += seconds * 1000

            tick_count = 0
            current_millisecond = 0
            sum_ask_price = 0
            sum_bid_price = 0
            sum_ask_size = 0
            sum_bid_size = 0



    return avg_data

# ---------------------------------------------------------------------------------------------------
# END INDICATORS
# ---------------------------------------------------------------------------------------------------


def load_data_from_csv(csv_path, limit):
    history_ts = pd.read_csv(csv_path, nrows=limit)
    history_ts.dropna(inplace=True)
    history_ts = history_ts[history_ts["bidPrice"] != 0]
    history_ts = history_ts[history_ts["askPrice"] != 0]
    history_ts.reset_index(inplace=True)

    # Compute mid and smart price - optional
    # smart_price_list = list()
    # mid_price_list = list()
    #
    # for index, row in history_ts.iterrows():
    #     bidPrice = float(row["bidPrice"])
    #     bidSize = float(row["bidSize"])
    #     askPrice = float(row["askPrice"])
    #     askSize = float(row["askSize"])
    #
    #     weight = askSize / (bidSize + askSize)
    #
    #     mid_price = (askPrice + bidPrice) / 2
    #     smart_price = (askPrice * weight) + (bidPrice * (1 - weight))
    #
    #     mid_price_list.append(mid_price)
    #     smart_price_list.append(smart_price)
    #
    # history_ts["midPrice"] = np.array(mid_price_list)
    # history_ts["smartPrice"] = np.array(smart_price_list)

    return history_ts

def average(lst):
    return sum(lst) / len(lst)

def get_lists_similarity(a, b):
    a_len = len(a)
    b_len = len(b)
    if a_len != b_len:
        print ("ERROR::gget_lists_similarity - bad length")
        return

    a_mean = average(a)
    b_mean = average(b)
    a_b_offset = b_mean - a_mean

    similarity = 0
    for i in range(a_len):
        b_offseted = b[i] - a_b_offset
        delta = a[i] - b_offseted
        similarity += abs(delta)

    return similarity

def get_chunks_similarity(a, b):
    ask_price_similarity = get_lists_similarity(a["askPrice"].data, b["askPrice"].data)
    return ask_price_similarity

def match_chunk_over_base(base, chunk, skip_begin, skip_end, plot = False):

    similarity_evolution = pd.DataFrame({"begin_index": list(), "similarity": list()})
    similarity_evolution_len = 0

    base_len = len(base.index)
    chunk_len = len(chunk.index)

    best_sim = 100000
    best_sim_start_index = -1

    for start_index in range(0, base_len - chunk_len):
        if skip_begin <= start_index < skip_end:
            similarity_evolution.loc[similarity_evolution_len] = [start_index, 1]
            similarity_evolution_len += 1
            continue

        end_index = start_index + chunk_len
        current_chunk = base[start_index:end_index]

        current_similarity = get_chunks_similarity(current_chunk, chunk)

        current_similarity /= chunk_len

        if current_similarity < best_sim:
            best_sim = current_similarity
            best_sim_start_index = start_index

        similarity_evolution.loc[similarity_evolution_len] = [start_index, current_similarity]
        similarity_evolution_len += 1

    for i in range(base_len - chunk_len, base_len):
        similarity_evolution.loc[similarity_evolution_len] = [i, 1]
        similarity_evolution_len += 1

    if plot:
        fig, axs = plt.subplots(2)
        axs[0].plot(base["second"], base["askPrice"], "-b")
        axs[0].plot(chunk["second"], chunk["askPrice"], "-r")
        axs[0].plot(base["second"][best_sim_start_index:best_sim_start_index + chunk_len], base["askPrice"][best_sim_start_index:best_sim_start_index + chunk_len], "--c")
        axs[1].plot(similarity_evolution["begin_index"], similarity_evolution["similarity"], "-m")
        plt.show()

    return best_sim, best_sim_start_index


def find_best_similarity(data):
    data_len = len(data.index)

    best_result = (10000, -1)

    similarity_evolution = list()

    chunk_len = 200
    evaluation_step = 10

    for i in range(0, data_len - chunk_len - 1, evaluation_step):
        res = match_chunk_over_base(data, data[i:i + chunk_len], i - chunk_len, i + 2 * chunk_len)
        if res[0] < best_result[0]:
            best_result = res
            print ("best sim: {}  {}   {} ".format(best_result[0], i, best_result[1]))

        similarity_evolution.append(res[0])

    plt.plot(similarity_evolution, "*m")
    plt.show()

    print ("ready")


def run(csv_path):
    tick_data = load_data_from_csv(csv_path, 10000)
    # tick_data[["askPrice"]].plot()
    # tick_data[["askSize", "bidSize"]].plot()

    averaged_data = moving_average_one_sec(tick_data, 1)

    # plt.plot(tick_data["timestamp"], tick_data["askPrice"], "-b")
    # plt.plot(averaged_data["timestamp"], averaged_data["askPrice"], "-r")
    # plt.show()

    # plt.plot(tick_data["timestamp"], tick_data["askPrice"], "-b")
    # plt.plot(averaged_data["second"], averaged_data["askPrice"], "-r")
    # plt.show()

    match_chunk_over_base(averaged_data, averaged_data[1220:1420], 1420, 1600, True)
    # match_chunk_over_base(averaged_data, averaged_data[1200:1300], 1100, 1400, True)
    # match_chunk_over_base(averaged_data, averaged_data[1200:1300], 1100, 1400, True)

    # find_best_similarity(averaged_data)

    print("Done")