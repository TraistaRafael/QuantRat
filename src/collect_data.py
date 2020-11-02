import os
import time
from binance.client import Client
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor

def start_data_collect() :

    api_key = "1w6kRRV7DqhowMhR4ePhdFwDSZxlXRSdMhdMcKJxcRFgbpvqARLMruTmiubRNbFH"
    api_secret = "NxBmZ7hL1ZB5QcXhsT7BDHF7AqEQlJXmRtccwSoftXmuCIm3S0JbB3xahiYUrE76"

    client = Client(api_key, api_secret)
    # client.API_URL = "https://testnet.binance.vision/api"

    out_file_name = "record_ETHUSDT_" + str(time.time()) + ".csv"

    with open(out_file_name, "a") as myfile:
        myfile.write("timestamp,bidPrice,bidSize,askPrice,askSize\n")

    def btc_trade_history(msg):
        if msg['e'] != 'error':
            with open(out_file_name, "a") as myfile:
                myfile.write("{},{},{},{},{}\n".format(msg['E'], msg['b'], msg['B'], msg['a'], msg['A']))

    # init and start the WebSocket
    bsm = BinanceSocketManager(client)
    conn_key = bsm.start_symbol_ticker_socket('ETHUSDT', btc_trade_history)
    bsm.start()

start_data_collect()