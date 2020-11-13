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

def vaca_strategy(csv_path) :
    market_data = load_data_from_csv(csv_path)
    # market_data[["askPrice", "bidPrice", "midPrice"]].plot()


    model_training_len = 100
    model_testing_len = 20
    model_training_step = 50

    processed_data = process_timeseries_for_training(market_data, model_training_len, model_testing_len, model_training_step)

    print("model_training_len: {}, model_testing_len: {}, model_training_step: {}   samples: {}".format(
        model_training_len,
        model_testing_len,
        model_training_step,
        len(processed_data.index)))

    processed_data_len = len(processed_data.index)
    subset_length = 150

    subset_start_index = 0

    ada_accuracy_avg = 0
    rf_accuracy_avg = 0
    test_count = 0

    investment = 0
    profit = 0

    investment_evolution = list()
    profit_evolution = list()

    processed_data.reset_index(inplace=True)

    success_predictions = list()
    fail_predictions = list()
    accuracy_evolution = list()

    while subset_start_index + subset_length < processed_data_len: # and subset_start_index < 20:

        processed_data_subset = processed_data[subset_start_index:subset_start_index + subset_length]
        processed_data_subset.reset_index(inplace=True)

        total_samples = len(processed_data_subset.index)
        train_count = total_samples - 1  # use the last item only for prediction

        train_ts = processed_data_subset[processed_data_subset.index < train_count]
        test_ts = processed_data_subset[processed_data_subset.index >= train_count]

        X_train = train_ts[["Stdev", "Variance", "Slope", "Stderr", "Movement"]]
        y_train = list(train_ts["Response"])

        X_test = test_ts[["Stdev", "Variance", "Slope", "Stderr", "Movement"]]
        y_test = list(test_ts["Response"])

        clf_rf = RandomForestClassifier()
        clf_rf.fit(X_train.values, y_train)

        # ada_score = clf_ada.score(X_test.values, y_test)
        rf_score = clf_rf.score(X_test.values, y_test)

        tr_begin_index = int(processed_data_subset.iloc[subset_length - 1]["PredictionBeginIndex"])
        tr_end_index = int(processed_data_subset.iloc[subset_length - 1]["PredictionEndIndex"])

        begin_ask_price = market_data.iloc[tr_begin_index]["askPrice"]
        begin_bid_price = market_data.iloc[tr_begin_index]["bidPrice"]
        end_ask_price = market_data.iloc[tr_end_index]["askPrice"]
        end_bid_price = market_data.iloc[tr_end_index]["bidPrice"]

        if rf_score == 1:
            success_predictions.append(1)
            fail_predictions.append(0)
            profit += end_bid_price - begin_ask_price
            profit_evolution.append(profit)
        else:
            success_predictions.append(0)
            fail_predictions.append(1)
            profit += begin_bid_price - end_ask_price
            profit_evolution.append(profit)

        # ada_accuracy_avg += ada_score
        rf_accuracy_avg += rf_score
        test_count += 1

        accuracy_evolution.append(float(rf_accuracy_avg / test_count))

        # print("ada: {}   rf: {}".format(ada_score, rf_score))
        print("{}  score: {}  profit: {}".format(subset_start_index, rf_accuracy_avg / test_count, profit))

        subset_start_index += 1

        # if subset_length > 50:
        # subset_length += 1

    # ada_accuracy_avg /= test_count
    rf_accuracy_avg /= test_count


    prediction_indices = list([i for i in range(test_count)])
    plt.plot(prediction_indices, profit_evolution, "-c")
    plt.plot(prediction_indices, accuracy_evolution, "-m")
    plt.show()

    print("final accuracy: ada: {}   rf: {}".format(ada_accuracy_avg, rf_accuracy_avg))
    print("final profit: {}".format(profit))

    # prediction_indices = list([i for i in range(test_count)])
    # plt.plot(prediction_indices, success_predictions, "*g")
    # plt.plot(prediction_indices, fail_predictions, "*r")
    # plt.plot(prediction_indices, accuracy_evolution, "-m")
    # plt.show()

# ---------------------------------------------------------------------------------------------------
def load_data_from_csv(csv_path):
    history_ts = pd.read_csv(csv_path)
    history_ts.dropna(inplace=True)
    history_ts = history_ts[history_ts["bidPrice"] != 0]
    history_ts = history_ts[history_ts["askPrice"] != 0]
    # history_ts = history_ts[history_ts["weightedPrice"] != 0]
    history_ts.reset_index(inplace=True)

    print("TradingEngine set intial market data len: {}".format(len(history_ts.index)))

    # train_test_ratio = 0.7
    # train_test_cut = int(len(history_ts.index) * train_test_ratio)
    # future_ts = history_ts[history_ts.index >= train_test_cut]
    # history_ts = history_ts[history_ts.index < train_test_cut]
    # history_ts.reset_index(inplace=True)
    # future_ts.reset_index(inplace=True)

    # Compute mid and smart price
    smart_price_list = list()
    mid_price_list = list()

    for index, row in history_ts.iterrows():
        bidPrice = float(row["bidPrice"])
        bidSize = float(row["bidSize"])
        askPrice = float(row["askPrice"])
        askSize = float(row["askSize"])

        weight = askSize / (bidSize + askSize)

        mid_price = (askPrice + bidPrice) / 2
        smart_price = (askPrice * weight) + (bidPrice * (1 - weight))

        mid_price_list.append(mid_price)
        smart_price_list.append(smart_price)

    history_ts["midPrice"] = np.array(mid_price_list)
    history_ts["smartPrice"] = np.array(smart_price_list)

    return history_ts




# ---------------------------------------------------------------------------------------------------
def process_timeseries_for_training(data, training_len, testing_len, training_step):
    '''
    Process raw tick timeseries
    @returns pandas timeseries
    Features: stdev, variance, slope, stderr, movement, response
    Response: 0, 1, -1 . Neutral, Up, Down
    '''

    data_len = len(data.index)

    processed_training_data = dict()
    processed_training_data["Stdev"] = list()
    processed_training_data["Variance"] = list()
    processed_training_data["Slope"] = list()
    processed_training_data["Stderr"] = list()
    processed_training_data["Movement"] = list()
    processed_training_data["Response"] = list()

    #aux
    processed_training_data["PredictionBeginIndex"] = list()
    processed_training_data["PredictionEndIndex"] = list()

    limit = data_len - training_len - testing_len - 1
    # limit = 2 *  training_len

    hold_count = 0
    up_count = 0
    down_count = 0

    timeframe_indices = list([i for i in range(training_len)])

    for index in range(0, limit, training_step):

        timeframe_mid_prices = list(data["midPrice"][index:index + training_len])

        # compute features values
        # mean = stat.mean(timeframe_mid_prices)
        stddev = stat.stdev(timeframe_mid_prices)
        variance = stat.variance(timeframe_mid_prices)
        linregress_obj = linregress(timeframe_indices, timeframe_mid_prices)
        training_movement = timeframe_mid_prices[-1] - timeframe_mid_prices[0]

        # compute response value
        openAskPrice = data["askPrice"][index + training_len]
        openBidPrice = data["bidPrice"][index + training_len]
        openMidPrice = data["midPrice"][index + training_len]

        closeAskPrice = 0
        closeBidPrice = 0

        up_down_response = 0

        closeAskPrice = data["askPrice"][index + training_len + testing_len]
        closeBidPrice = data["bidPrice"][index + training_len + testing_len]
        closeMidPrice = data["midPrice"][index + training_len + testing_len]

        if closeBidPrice > openAskPrice:
            up_down_response = 1  # Up
            up_count += 1
        elif closeAskPrice < openBidPrice:
            up_down_response = -1  # Down
            down_count += 1
        else:
            up_down_response = 0  # Hold
            hold_count += 1

        # figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
        # plt.plot(data.index[index:index + training_len], data["bidPrice"][index:index + training_len], "--b",
        #          label="bidPrice")
        # plt.plot(data.index[index:index + training_len], data["askPrice"][index:index + training_len], "-b",
        #          label="askPrice")
        # plt.plot(data.index[index:index + training_len + testing_len],
        #          data["midPrice"][index:index + training_len + testing_len], "--g", label="midPrice")
        # plt.plot(data.index[index:index + training_len + testing_len],
        #          data["smartPrice"][index:index + training_len + testing_len], "--r", label="smartPrice")
        # plt.plot(data.index[index + training_len:index + training_len + testing_len],
        #          data["askPrice"][index + training_len:index + training_len + testing_len], "--r",
        #          label="askPrice after")
        # plt.plot(data.index[index + training_len:index + training_len + testing_len],
        #          data["bidPrice"][index + training_len:index + training_len + testing_len], "-r",
        #          label="bidPrice after")
        # plt.plot([data.index[index + training_len]], [data["bidPrice"][index + training_len]], "*g",
        #          label="Open time")
        # plt.figtext(.2, 0.8, up_down_response)
        # plt.legend(loc="upper right")
        # plt.show()

        processed_training_data["Stdev"].append(stddev)
        processed_training_data["Variance"].append(variance)
        processed_training_data["Slope"].append(linregress_obj.slope)
        processed_training_data["Stderr"].append(linregress_obj.stderr)
        processed_training_data["Movement"].append(training_movement)
        processed_training_data["Response"].append(up_down_response)

        #aux
        processed_training_data["PredictionBeginIndex"].append(index)
        processed_training_data["PredictionEndIndex"].append(index + training_len)

    return pd.DataFrame.from_dict(processed_training_data)

