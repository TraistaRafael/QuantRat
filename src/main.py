'''
Copyright 2020 by Traista Viorel-Rafael
All rights reserved.
This file is part of the Traista Viorel-Rafael - QuantRat project
https://github.com/TraistaRafael/QuantRat
Please see the LICENSE.txt
'''

from trading_engine_1 import ReTradingEngine
import Trading
import vaca_strategy_dev
import cainele_alb
import calul_mov
import dealu_valea

# vaca_strategy_dev.vaca_strategy("C:/Projects/QuantRat/data/ethusd_3_nov.csv")
# cainele_alb.run("C:/Projects/QuantRat/data/record_ETHUSDT_3_mb.csv")
# calul_mov.run()
dealu_valea.run()

# Trading.run_simulator()

# engine = ReTradingEngine()
# engine.re_training_strategy_test("C:/Projects/QuantRat/data/btc_full.csv")
# engine.re_training_strategy_test_fixed_size("C:/Projects/QuantRat/data/btc_full.csv")
# engine.re_training_strategy_test_fixed_size("C:/Projects/QuantRat/data/ethusd_sat_full.csv")
# engine.set_initial_data_from_csv("C:/Projects/QuantRat/data/btc_train.csv")
# engine.optimize_model_params()
# engine.re_training_strategy_test("C:/Projects/QuantRat/data/btc_full.csv", 10000, 250)
#
#

