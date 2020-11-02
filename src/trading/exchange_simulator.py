'''
Copyright 2020 by Traista Viorel-Rafael
All rights reserved.
This file is part of the Traista Viorel-Rafael - QuantRat project
https://github.com/TraistaRafael/QuantRat
Please see the LICENSE.txt
'''

from trading_engine import TradingEngine

class ExchangeSimulator :

    def __init__(self) :
        print ("ExchangeSimulator __init__()")
        self.engine = TradingEngine()
        self.running = False


    def simulate_0(self) :
        print ("ExchangeSimulator simulate_0()")
        # init the trading engine
        self.engine.set_initial_data_from_csv("C:/Projects/QuantRat/data/btc_train.csv")
        # self.engine.optimize_model_params()
        self.engine.train()

        self.engine.print_current_acuracy(0.95)

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