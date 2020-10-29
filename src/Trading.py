'''
Copyright 2020 by Traista Viorel-Rafael
All rights reserved.
This file is part of the Traista Viorel-Rafael - QuantRat project
https://github.com/TraistaRafael/QuantRat
Please see the LICENSE.txt
'''

import FinanceUtils

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier)
import matplotlib.pyplot as plt
import statistics as stat
from scipy.stats import linregress
from matplotlib.pyplot import figure
import time
import random

class TradingEngine :
    '''!
    @brief short description...
    longer description
    '''

    def __init__(self) :
        self.market_data_locked = False
        self.market_data_changed = False
        self.market_data = pd.DataFrame()

        self.model_training_len = 800
        self.model_testing_len = 200
        self.model_training_step = 100
        
        print("TradingEngine __init__()")


    def lock_data(self) :
        '''
        @brief Protect self.market_data with mutex, because this will be possible used in multiple threads
        '''
        while self.market_data_locked == True :
            time.sleep(0.1) #Wait for a few millis   
        
        self.market_data_locked = True


    def unlock_data(self) :
        self.market_data_locked = False


    def set_initial_data_from_csv(self, csv_path) :
        print("TradingEngine set_initial_data()")

        history_ts = pd.read_csv(csv_path)
        history_ts.dropna(inplace=True)
        history_ts = history_ts[history_ts["bidPrice"] != 0]
        history_ts = history_ts[history_ts["askPrice"] != 0]
        history_ts = history_ts[history_ts["weightedPrice"] != 0]
        history_ts.reset_index(inplace=True)

        train_test_ratio = 0.7
        train_test_cut = int(len(history_ts.index) * train_test_ratio)
        future_ts = history_ts[history_ts.index >= train_test_cut]
        history_ts = history_ts[history_ts.index < train_test_cut]
        history_ts.reset_index(inplace=True)
        future_ts.reset_index(inplace=True)

        #Compute mid and smart price
        smart_price_list = list()
        mid_price_list = list()

        for index, row in history_ts.iterrows() :
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

        self.lock_data()
        self.market_data = history_ts
        self.unlock_data()


    def train(self) :
        print("TradingEngine train()")
        processed_data = self.process_timeseries_for_training(self.market_data, self.model_training_len, self.model_testing_len, self.model_training_step)
        
        total_samples = len(processed_data.index)
        train_count = int(total_samples * train_test_ratio)
    
        train_ts = processed_data[processed_data.index < train_count]
        test_ts = processed_data[processed_data.index >= train_count]

        X_train = train_ts[["Stdev", "Variance", "Slope", "Stderr", "Movement"]]
        y_train = list(train_ts["Response"])

        X_test = test_ts[["Stdev", "Variance", "Slope", "Stderr", "Movement"]]
        y_test = list(test_ts["Response"])

        clf_ada = AdaBoostClassifier()
        clf_ada.fit(X_train.values, y_train)

        clf_rforest = RandomForestClassifier(n_estimators=100, min_samples_split=2)
        clf_rforest.fit(X_train.values, y_train)


    def on_market_tick(self, timestamp, symbol, bid_price, bid_size, ask_price, ask_size) :
        print("TradingEngine on_market_tick()")

        smart_price = FinanceUtils.get_smart_price(bid_price, bid_size, ask_price, ask_size)
        mid_price = FinanceUtils.get_mid_price(bid_price, ask_price)

        new_row = {
            'timestamp': [timestamp],
            'symbol': [symbol],
            'weightedPrice': [0],
            'bidPrice': [bid_price],
            'bidSize': [bid_size],
            'askPrice': [ask_price],
            'askSize': [ask_size],
            'midPrice': [mid_price],
            'smartPrice': [smart_price]
        }
 
        self.lock_data()
        #Append the new row to the existing dataframe
        self.market_data.append(pd.DataFrame.from_dict(new_row))
        self.unlock_data()


    def process_timeseries_for_training(self, data, training_len, testing_len, training_step) :
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

        limit = data_len - training_len - testing_len - 1
        limit = 2 *  training_len

        hold_count = 0
        up_count = 0
        down_count = 0

        timeframe_indices = list([i for i in range(training_len)])

        for index in range(0, limit, training_step) :
    
            timeframe_mid_prices = list(data["midPrice"][index:index + training_len])
            
            #compute features values
            # mean = stat.mean(timeframe_mid_prices)
            stddev = stat.stdev(timeframe_mid_prices)
            variance = stat.variance(timeframe_mid_prices)
            linregress_obj = linregress(timeframe_indices, timeframe_mid_prices)
            training_movement = timeframe_mid_prices[-1] - timeframe_mid_prices[0]

            #compute response value
            openAskPrice = data["askPrice"][index + training_len]
            openBidPrice = data["bidPrice"][index + training_len]
            openMidPrice = data["midPrice"][index + training_len]

            closeAskPrice = 0
            closeBidPrice = 0

            up_down_response = 0 
            
            closeAskPrice = data["askPrice"][index + training_len + testing_len]
            closeBidPrice = data["bidPrice"][index + training_len + testing_len]
            closeMidPrice = data["midPrice"][index + training_len + testing_len]

            if closeBidPrice > openAskPrice :
                up_down_response = 1 # Up
                up_count += 1
            elif closeAskPrice < openBidPrice :
                up_down_response = -1 # Down
                down_count += 1
            else :
                up_down_response = 0 # Hold
                hold_count += 1

            # figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
            # plt.plot(data.index[index:index + training_len], data["bidPrice"][index:index + training_len], "--b", label="bidPrice")
            # plt.plot(data.index[index:index + training_len], data["askPrice"][index:index + training_len], "-b", label="askPrice")
            # plt.plot(data.index[index:index + training_len + testing_len], data["midPrice"][index:index + training_len + testing_len], "--g", label="midPrice")
            # plt.plot(data.index[index:index + training_len + testing_len], data["smartPrice"][index:index + training_len + testing_len], "--r", label="smartPrice")
            # plt.plot(data.index[index + training_len:index + training_len + testing_len], data["askPrice"][index + training_len:index + training_len + testing_len], "--r", label="askPrice after")
            # plt.plot(data.index[index + training_len:index + training_len + testing_len], data["bidPrice"][index + training_len:index + training_len + testing_len], "-r", label="bidPrice after")
            # plt.plot([data.index[index+training_len]], [data["bidPrice"][index+training_len]], "*g", label="Open time")
            # plt.figtext(.2, 0.8, up_down_response)
            # plt.legend(loc="upper right")
            # plt.show()

            processed_training_data["Stdev"].append(stddev)
            processed_training_data["Variance"].append(variance)
            processed_training_data["Slope"].append(linregress_obj.slope)
            processed_training_data["Stderr"].append(linregress_obj.stderr)
            processed_training_data["Movement"].append(training_movement)
            processed_training_data["Response"].append(up_down_response)

        return pd.DataFrame.from_dict(processed_training_data)

    
    def get_strategy_acuracy_on_params(self, data, training_len, testing_len, training_step, train_test_ratio) :
        processed_data = self.process_timeseries_for_training(data, training_len, testing_len, training_step)
        
        total_samples = len(processed_data.index)
        train_count = int(total_samples * train_test_ratio)
    
        train_ts = processed_data[processed_data.index < train_count]
        test_ts = processed_data[processed_data.index >= train_count]

        X_train = train_ts[["Stdev", "Variance", "Slope", "Stderr", "Movement"]]
        y_train = list(train_ts["Response"])

        X_test = test_ts[["Stdev", "Variance", "Slope", "Stderr", "Movement"]]
        y_test = list(test_ts["Response"])

        clf_ada = AdaBoostClassifier()
        clf_ada.fit(X_train.values, y_train)

        clf_rforest = RandomForestClassifier(n_estimators=100, min_samples_split=2)
        clf_rforest.fit(X_train.values, y_train)

        return {
            "adaboost": "{:.2f}".format(clf_ada.score(X_test.values, y_test)),
            "random_forest": "{:.2f}".format(clf_rforest.score(X_test.values, y_test))
        }


    def optimize_model_params(self, data) :

        params_dict = {
            "training_len": [800, 900, 1000, 1100, 1200],
            "testing_len": [100, 150, 200, 300, 400],
        }

        training_step = 100
        train_test_ratio = 0.8

        param_num = len(params_dict["training_len"])
        combinations_num = param_num ** 2

        print ("Run {} parameter combinations".format(combinations_num))

        for i in range(combinations_num) :
            index_a = int(i % param_num)
            index_b = int(i / param_num)

            print ("{}, {}".format(index_b, index_a))

            training_len_p = params_dict["training_len"][index_a]
            testing_len_p = params_dict["testing_len"][index_b]

            fitness = self.get_strategy_acuracy_on_params(data, training_len_p, testing_len_p, training_step, train_test_ratio )
            
            print ("training_len: {}, testing_len: {}, acuracy: {}".format(training_len_p, testing_len_p, fitness))


    # fitness = get_strategy_acuracy_on_params(history_ts, 1000, 200, 50, 0.8)
    # print ("fitness: {}", fitness)

    def check_before_prediction(self) :
        return True
        print("TradingEngine check_before_prediction()")


    def check_after_prediction(self) :
        return True
        print("TradingEngine check_after_prediction()")


    def predict(self) :
        '''!
        @brief Predicts next market movement on some period.
        @returns Integer in {-1, 0, 1} meaning down, neutral, up. 
        '''
        print("TradingEngine predict()")

        training_len = 900
        hist_len = len(self.market_data.index)

        #subset data
        observed_indices = list([i for i in range(training_len)])
        observed_mid_prices = list(history_ts["midPrice"][hist_len - training_len:])

        stddev = stat.stdev(observed_mid_prices)
        variance = stat.variance(observed_mid_prices)
        linregress_obj = linregress(observed_indices, observed_mid_prices)
        training_movement = observed_mid_prices[-1] - observed_mid_prices[0]


    def commit_on_market(self, action) :
        print("TradingEngine commit_on_market()")



# # %%



    



# # optimize_model_params(history_ts)

# # %%
# def on_market_tick(timestamp, symbol, bid_price, bid_size, ask_price, ask_size) :
#     smart_price = get_smart_price(bid_price, bid_size, ask_price, ask_size)
#     mid_price = get_mid_price(bid_price, ask_price)

#     new_row = {
#         'timestamp': timestamp,
#         'symbol': symbol,
#         'weightedPrice': 0,
#         'bidPrice': bid_price,
#         'bidSize': bid_size,
#         'askPrice': ask_price,
#         'askSize': ask_size,
#         'midPrice': mid_price,
#         'smartPrice': smart_price
#     }

# # %%
# def predict_movement_strategy_0(history_ts) :

#     training_len = 900
#     hist_len = len(history_ts.index)

#     #subset data
#     observed_indices = list([i for i in range(training_len)])
#     observed_mid_prices = list(history_ts["midPrice"][hist_len - training_len:])

#     stddev = stat.stdev(observed_mid_prices)
#     variance = stat.variance(observed_mid_prices)
#     linregress_obj = linregress(observed_indices, observed_mid_prices)
#     training_movement = observed_mid_prices[-1] - observed_mid_prices[0]
