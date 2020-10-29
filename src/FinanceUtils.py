'''
Copyright 2020 by Traista Viorel-Rafael
All rights reserved.
This file is part of the Traista Viorel-Rafael - QuantRat project
https://github.com/TraistaRafael/QuantRat
Please see the LICENSE.txt
'''

def get_smart_price(self, bid_price, bid_size, ask_price, ask_size) :
    weight = ask_size / (bid_size + ask_size)
    smart_price = (ask_price * weight) + (bid_price * (1 - weight))
    return smart_price


def get_mid_price(self, bid_price, ask_price) :
    return (ask_price + bid_price) / 2
