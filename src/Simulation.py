'''
Copyright 2020 by Traista Viorel-Rafael
All rights reserved.
This file is part of the Traista Viorel-Rafael - QuantRat project
https://github.com/TraistaRafael/QuantRat
Please see the LICENSE.txt
'''

import TradingEngine

class ExchangeSimulator :

    def __init__(self) :
        print ("ExchangeSimulator __init__()")
        self.TradingEngine = TradingEngine.TradingEngine()

    def init_trading_engine_0(self) :
        self.TradingEngine.set_initial_data(None)

    def start(self) :
        print ("ExchangeSimulator start()")

