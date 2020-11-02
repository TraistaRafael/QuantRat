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
from os import path

class TradingEngine :
    '''!
    @brief short description...
    longer description
    '''
    def __init__(self) :
        self.market_data_locked = False
        self.market_data_changed = False
        self.market_data = pd.DataFrame()
        self.initial_market_data_len = 0

        self.model_training_len = 900
        self.model_testing_len = 200
        self.model_training_step = 100
        self.model_train_test_ratio = 0.8

        self.model = None
        
        self.last_prediction_index = 0
        self.prediction_step = 100

        self.prediction_history = {
            "checked": list(),
            "market_data_start_index": list(),
            "market_data_stop_index": list(),
            "timestamp": list(),
            "prediction": list(),
            "actual": list(),
            "trend_success": list(),
            "profit_success": list(),
            "profit": list()
        }

        print("TradingEngine __init__()")

    #---------------------------------------------------------------------------------------------------
    def print_current_acuracy(self, train_test_ratio, training_step) :
        # processed_data = self.process_timeseries_for_training(self.market_data, self.model_training_len, self.model_testing_len, training_step)
        
        # total_samples = len(processed_data.index)
        # train_count = int(total_samples * train_test_ratio)
    
        # test_ts = processed_data[processed_data.index >= train_count]

        # X_test = test_ts[["Stdev", "Variance", "Slope", "Stderr", "Movement"]]
        # y_test = list(test_ts["Response"])

        score = self.get_strategy_acuracy_on_params(self.market_data, self.model_training_len, self.model_testing_len, self.model_training_step, train_test_ratio)
        print("print_current_acuracy on ratio: {:.0f}, score: {}".format(train_test_ratio, score))


    #---------------------------------------------------------------------------------------------------
    def lock_data(self) :
        '''
        @brief Protect self.market_data with mutex, because this will be possible used in multiple threads
        '''
        while self.market_data_locked == True :
            time.sleep(0.1) #Wait for a few millis   
        
        self.market_data_locked = True

    #---------------------------------------------------------------------------------------------------
    def unlock_data(self) :
        self.market_data_locked = False

    #---------------------------------------------------------------------------------------------------
    def set_initial_data_from_csv(self, csv_path) :
        print("TradingEngine set_initial_data()")

        history_ts = pd.read_csv(csv_path)
        history_ts.dropna(inplace=True)
        history_ts = history_ts[history_ts["bidPrice"] != 0]
        history_ts = history_ts[history_ts["askPrice"] != 0]
        history_ts = history_ts[history_ts["weightedPrice"] != 0]
        history_ts.reset_index(inplace=True)

        print ("TradingEngine set intial market data len: {}".format(len(history_ts.index)))

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

        self.initial_market_data_len = len(self.market_data.index)

    #---------------------------------------------------------------------------------------------------
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
        #limit = 2 *  training_len

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

    #--------------------------------------------------------------------------------------------------- 
    def get_strategy_acuracy_on_params(self, data, training_len, testing_len, training_step, train_test_ratio) :
        processed_data = self.process_timeseries_for_training(data, training_len, testing_len, training_step)

        print ("get strategy acuracy on params: {}".format(len(processed_data.index)))

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

    #---------------------------------------------------------------------------------------------------
    def optimize_model_params(self) :

        params_dict = {
            "training_len": [800, 900, 1000, 1100, 1200],
            "testing_len": [100, 150, 200, 300, 400],
        }

        training_step = 20
        train_test_ratio = 0.95

        param_num = len(params_dict["training_len"])
        combinations_num = param_num ** 2

        print ("Run {} parameter combinations".format(combinations_num))

        for i in range(combinations_num) :
            index_a = int(i % param_num)
            index_b = int(i / param_num)

            print ("{}, {}".format(index_b, index_a))

            training_len_p = params_dict["training_len"][index_a]
            testing_len_p = params_dict["testing_len"][index_b]

            self.market_data[["midPrice"]].plot()
            fitness = self.get_strategy_acuracy_on_params(self.market_data, training_len_p, testing_len_p, training_step, train_test_ratio )
            
            print ("training_len: {}, testing_len: {}, acuracy: {}".format(training_len_p, testing_len_p, fitness))

    #---------------------------------------------------------------------------------------------------
    def test_live_training_strategy(self, csv_path):

        train_start_index = 0;



    #---------------------------------------------------------------------------------------------------
    def check_before_prediction(self) :
        return True
        print("TradingEngine check_before_prediction()")

    #---------------------------------------------------------------------------------------------------
    def check_after_prediction(self) :
        return True
        print("TradingEngine check_after_prediction()")

    #---------------------------------------------------------------------------------------------------
    def train(self) :
        '''
        @brief regenerate the predictive model on existing market data, and using current optimized parameters
        '''

        print("TradingEngine train()")
        processed_data = self.process_timeseries_for_training(self.market_data, self.model_training_len, self.model_testing_len, self.model_training_step)
        
        X_train = processed_data[["Stdev", "Variance", "Slope", "Stderr", "Movement"]]
        y_train = list(processed_data["Response"])

        # clf_ada = AdaBoostClassifier()
        # clf_ada.fit(X_train.values, y_train)

        self.model = RandomForestClassifier(n_estimators=100, min_samples_split=2)
        self.model.fit(X_train.values, y_train)

    #---------------------------------------------------------------------------------------------------
    def test_model_on_csv(self, csv_path) :
        print("TradingEngine test_model_on_csv()")

        if self.model == None:
            print ("self.model == None")
            return

        test_market_data = pd.read_csv(csv_path)
        test_market_data.dropna(inplace=True)
        test_market_data = test_market_data[test_market_data["bidPrice"] != 0]
        test_market_data = test_market_data[test_market_data["askPrice"] != 0]
        test_market_data = test_market_data[test_market_data["weightedPrice"] != 0]
        test_market_data.reset_index(inplace=True)

        print ("TradingEngine test on market data len: {}".format(len(test_market_data.index)))

        #Compute mid and smart price
        smart_price_list = list()
        mid_price_list = list()

        for index, row in test_market_data.iterrows() :
            bidPrice = float(row["bidPrice"])
            bidSize = float(row["bidSize"])
            askPrice = float(row["askPrice"])
            askSize = float(row["askSize"])

            weight = askSize / (bidSize + askSize)
            
            mid_price = (askPrice + bidPrice) / 2
            smart_price = (askPrice * weight) + (bidPrice * (1 - weight))

            mid_price_list.append(mid_price)
            smart_price_list.append(smart_price)

        test_market_data["midPrice"] = np.array(mid_price_list)
        test_market_data["smartPrice"] = np.array(smart_price_list)


        #prepared data for the model and predict
        processed_data = self.process_timeseries_for_training(test_market_data, self.model_training_len, self.model_testing_len, self.model_training_step)
        
        X_test = processed_data[["Stdev", "Variance", "Slope", "Stderr", "Movement"]]
        y_test = list(processed_data["Response"])

        print ("Test model on csv acuracy: {}".format(self.model.score(X_test.values, y_test)))

    #---------------------------------------------------------------------------------------------------
    def on_market_tick(self, timestamp, symbol, bid_price, bid_size, ask_price, ask_size) :
        '''!
        @brief add new market info row
        '''
        #print("TradingEngine on_market_tick()")

        smart_price = FinanceUtils.get_smart_price(bid_price, bid_size, ask_price, ask_size)
        mid_price = FinanceUtils.get_mid_price(bid_price, ask_price)

        self.lock_data()
        #Append the new row to the existing dataframe

        last_index = len(self.market_data.index)
        # row_len = len(new_row_dict.index)

        # print(str(self.market_data.tail()))
        # self.market_data = self.market_data.append(new_row_dict, sort=False, inplace=True)
        #self.market_data = self.market_data.append(new_row_dict, sort=False)
        #self.market_data.reset_index(inplace=True)

        self.market_data.loc[last_index] = [last_index, timestamp, symbol, 0, bid_price, bid_size, ask_price, ask_size, mid_price, smart_price]

        # print(str(self.market_data.tail()))

        # hist_len = len(self.market_data.index)

        self.unlock_data()

        if False and self.last_prediction_index > self.prediction_step :
            print ("market_data length: {}".format(len(self.market_data.index)))
            prediction = self.predict()         
            if prediction != None :
                self.commit_on_history(timestamp, prediction)
            self.last_prediction_index = 0
        else :
            self.last_prediction_index += 1

    #---------------------------------------------------------------------------------------------------
    def predict(self) :
        '''!
        @brief Predicts next market movement on some period.
        @returns Integer in {-1, 0, 1} meaning down, neutral, up. 
        '''
        #print("TradingEngine predict()")

        data_len = len(self.market_data.index)

        #subset data
        observed_indices = list([i for i in range(self.model_training_len)])
        observed_mid_prices = list(self.market_data["midPrice"][data_len - self.model_training_len:])

        stddev = stat.stdev(observed_mid_prices)
        variance = stat.variance(observed_mid_prices)
        linregress_obj = linregress(observed_indices, observed_mid_prices)
        training_movement = observed_mid_prices[-1] - observed_mid_prices[0]

        prediction = self.model.predict([[stddev, variance, linregress_obj.slope, linregress_obj.stderr, training_movement]])
        if len(prediction) == 0 :
            print ("TrainingEngine Prediction failed")
            return None
        else:
            # print ("prediction: {},  {}, {}, {}, {} --> {}".format(stddev, variance, linregress_obj.slope, linregress_obj.stderr, training_movement, prediction[0]))
            return prediction[0]
 
    #---------------------------------------------------------------------------------------------------
    def commit_on_history(self, timestamp, prediction) :
    
        #check previous prediction
        self.lock_data()
    
        hist_len = len(self.prediction_history["timestamp"])
        market_data_len = len(self.market_data.index)

        if (hist_len > 0) :
            for i in range (hist_len) :
                if self.prediction_history["checked"][i] == False :
                    
                    prev_pred_timestamp = self.prediction_history["timestamp"][i]
                    prev_pred_start_index =  self.prediction_history["market_data_start_index"][i]
                    prev_pred_stop_index =  self.prediction_history["market_data_stop_index"][i]
                    prev_pred = self.prediction_history["prediction"][i]

                    if (prev_pred_stop_index >= market_data_len):
                        continue

                    prev_pred_ask_price = self.market_data["askPrice"][prev_pred_start_index]
                    prev_pred_mid_price = self.market_data["midPrice"][prev_pred_start_index]
                    prev_pred_bid_price = self.market_data["bidPrice"][prev_pred_start_index]
                    current_ask_price = self.market_data["askPrice"][prev_pred_stop_index]
                    current_mid_price = self.market_data["midPrice"][prev_pred_stop_index]
                    current_bid_price = self.market_data["bidPrice"][prev_pred_stop_index]

                    # delta = current_mid_price - prev_pred_mid_price
                    # trend_success = False
                    # profit_success = False
                    # profit = 0
                    
                    pred_response = 0 #hold
                    if current_bid_price > prev_pred_ask_price :
                        pred_response = 1 # Up
                    elif current_ask_price < prev_pred_bid_price :
                        pred_response = -1 # Down
                    else :
                        pred_response = 0 # Hold

                    self.prediction_history["actual"][i] = pred_response
                    self.prediction_history["checked"][i] = True

                    # if prev_pred > 0 :
                    #     #previous prediction was up
                    #     if delta > 0 :
                    #         trend_success = True
                    #     if current_bid_price > prev_pred  _ask_price :
                    #         profit_success = True
                    #     profit = current_bid_price - prev_pred_ask_price
                    # else :
                    #     #previous prediction was down
                    #      if delta < 0 :
                    #         trend_success = True
                    #     if current_ask_price < prev_pred_bid_price :
                    #         profit_success = True
                    #     profit = current_ask_price - prev_pred_bid_price
              
        self.unlock_data()

        #add new history entry
        self.prediction_history["market_data_start_index"].append(market_data_len)
        self.prediction_history["market_data_stop_index"].append(market_data_len + self.model_testing_len)
        self.prediction_history["timestamp"].append(timestamp)
        self.prediction_history["prediction"].append(prediction)
        self.prediction_history["actual"].append(prediction)
        self.prediction_history["trend_success"].append(False)
        self.prediction_history["profit_success"].append(False)
        self.prediction_history["profit"].append(0)
        self.prediction_history["checked"].append(False)

    #---------------------------------------------------------------------------------------------------
    def commit_on_market(self, action) :
        print("TradingEngine commit_on_market()")

    #---------------------------------------------------------------------------------------------------
    def plot_prediction_history(self) :
        
        total_predictions = len(self.prediction_history["prediction"])
        correct_predictions = 0

        for i in range(total_predictions) :
            if self.prediction_history["prediction"][i] == self.prediction_history["actual"][i] :
                correct_predictions += 1
        
        print("correct: {}, total: {}, accuracy: {}".format(correct_predictions, total_predictions, correct_predictions/total_predictions))

        plt.plot(self.prediction_history["prediction"], "*r", label="Prediction")
        plt.plot(self.prediction_history["actual"], "*g", label="Response")
        plt.show()

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

    #---------------------------------------------------------------------------------------------------
    def plot_market_data(self) :
        plt.plot(self.market_data.index[0:self.initial_market_data_len], self.market_data["bidPrice"][0:self.initial_market_data_len], "-b", label="initial bidPrice")
        plt.plot(self.market_data.index[0:self.initial_market_data_len], self.market_data["askPrice"][0:self.initial_market_data_len], "--b", label="initial bidPrice")

        plt.plot(self.market_data.index[self.initial_market_data_len:], self.market_data["bidPrice"][self.initial_market_data_len:], "-r", label="tracked bidPrice")
        plt.plot(self.market_data.index[self.initial_market_data_len:], self.market_data["askPrice"][self.initial_market_data_len:], "--r", label="tracked bidPrice")

        plt.show()


class ExchangeSimulator :


    def __init__(self) :
        print ("ExchangeSimulator __init__()")
        self.engine = TradingEngine()
        self.running = False


    def simulate_0(self) :
        print ("ExchangeSimulator simulate_0()")
        # init the trading engine
        self.engine.set_initial_data_from_csv("C:/Projects/QuantRat/data/btc_train.csv")
        self.engine.optimize_model_params()
        self.engine.train()

        self.engine.print_current_acuracy(0.92, 50)

        self.engine.test_model_on_csv("C:/Projects/QuantRat/data/btc_test.csv")

        return
        # fake exchange data stream
        train_data = pd.read_csv("C:/Projects/QuantRat/data/btc_test.csv")
        train_data.dropna(inplace=True)
        train_data = train_data[train_data["bidPrice"] != 0]
        train_data = train_data[train_data["askPrice"] != 0]
        train_data = train_data[train_data["weightedPrice"] != 0]
        train_data.reset_index(inplace=True)


        print("ExchangeSimulator simulating...")

        for index, row in train_data.iterrows() :

            timestamp = float(row["timestamp"])
            bid_price = float(row["bidPrice"])
            bid_size = float(row["bidSize"])
            ask_price = float(row["askPrice"])
            ask_size = float(row["askSize"])

            # timestamp, symbol, bid_price, bid_size, ask_price, ask_size
            self.engine.on_market_tick(timestamp, "ETHUSDT", bid_price, bid_size, ask_price, ask_size)

        print("ExchangeSimulator ssimulation done")
        self.engine.plot_market_data()
        # self.engine.plot_prediction_history()

def run_simulator() :
    simulator = ExchangeSimulator()
    simulator.simulate_0()


# run_simulator()