'''
Copyright 2020 by Traista Viorel-Rafael
All rights reserved.
This file is part of the Traista Viorel-Rafael - QuantRat project
https://github.com/TraistaRafael/QuantRat
Please see the LICENSE.txt
'''

class TradingEngine :
    '''!
    @brief short description...
    longer description
    '''

    def __init__(self):
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
