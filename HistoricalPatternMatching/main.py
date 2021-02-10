'''
Copyright (c) 2021 Traista Rafael

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See LICENSE.TXT
'''

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# https://gist.github.com/alberto-santini/9de98621f85cb5036756d6d379d95949
def flatten(a):
    return sum(a, [])

def local_minima(a):
    if len(a) < 2:
        return list(range(len(a)))
    
    lmin = True
    curr = list()
    local_minima = list()
    
    for i, x in enumerate(a[:-1]):
        if a[i + 1] == x and lmin:
            curr.append(i)
        if a[i + 1] < x:
            lmin = True
            curr = list()
        if a[i + 1] > x:
            if lmin:
                local_minima.append(curr + [i])
            lmin = False

    if a[-1] < a[-2] or (lmin and a[-1] == a[-2]):
        local_minima.append(curr + [len(a) - 1])
    
    return flatten(local_minima)

def invert(a):
    return [-x for x in a]

def local_maxima(a):
    return local_minima(invert(a))
    

class HistoricalPatternMatching :

    def __init__(self, _ticker, _period, _tick_interval, csv_path=None) :

        if csv_path is None:
            self.data = data = yf.download(
                tickers=_ticker,
                period = _period,
                interval = _tick_interval,
                group_by = 'ticker',
                auto_adjust = True,
                prepost = True,
                threads = True,
                proxy = None)
            # self.data.to_csv("1_week_30min.csv")
        else:
            self.data = pd.read_csv(csv_path)       
       

    def get_data(self):
        return self.data


    def average(self, lst):
        return sum(lst) / len(lst)


    def get_lists_similarity(self, a, b):
        a_len = len(a)
        b_len = len(b)
        if a_len != b_len:
            print ("ERROR::get_lists_similarity - bad length")
            return

        a_mean = self.average(a)
        b_mean = self.average(b)
        a_b_offset = b_mean - a_mean

        similarity = 0
        a_mean_offset = 0
        for i in range(a_len):
            b_offseted = b[i] - a_b_offset
            delta = a[i] - b_offseted
            similarity += abs(delta)
            a_mean_offset += abs(a[i] - a_mean)

        a_mean_offset *= 2
        return similarity / a_mean_offset


    def fragments_similarity(self, a_begin, a_end, b_begin, b_end, columns=["Open", "High", "Low", "Close"]):
        similarity = 0
        
        for column in columns:
            similarity += self.get_lists_similarity(list(self.data[column][a_begin:a_end]), list(self.data[column][b_begin:b_end])) 

        similarity = similarity / len(columns)

        return similarity


    def moving_frame_similarity(self, frame_begin, frame_end, columns=["Open", "High", "Low", "Close"]):
        similarity_history = list()
        frame_len = frame_end - frame_begin

        data_len = len(self.data.index)
        for i in range(0, data_len):
            if i < frame_begin:
                local_similarity = self.fragments_similarity(i, i+frame_len, frame_begin, frame_end, columns)
                similarity_history.append(local_similarity)
            else:
                similarity_history.append(0)

        return similarity_history

    
    def plot_similarity_history(self, history, columns):

        # Extract local minima
        history_local_minima = local_minima(history)

        _, ax = plt.subplots(nrows=2, ncols=1, figsize=(16,8))
        
        ax[0].set_title('Price')
        for column in columns:
            ax[0].plot(self.data[column])

        ax[1].set_title('Similarity')
        ax[1].plot(history)
        ax[1].plot(history_local_minima, list([history[i] for i in history_local_minima]), "r*")
        plt.show()

# Unit tests
def test_lists_similarity():
    hist = HistoricalPatternMatching("EURUSD=X", "1d", "30m", csv_path="test_data.csv")
    
    assert hist.get_lists_similarity([0, -1, -2, -2, 0, 1, 2, 3, 3, 2], [0, -1, -2, -2, 0, 1, 2, 3, 3, 2]) == 0
    assert hist.get_lists_similarity([0, -1, -2, -2, 0, 1, 2, 3, 3, 2], [0, 1, 2, 2, 0, -1, -2, -3, -3, -2]) == 1

    similarity = hist.get_lists_similarity([1, -1, -2, -2, 0, 1, 2, 3, 3, 2], [0, -1, -2, -2, 0, 1, 2, 3, 3, 2])
    assert similarity > 0.057 and similarity < 0.058 


def test_moving_frame_similarity():
    hist = HistoricalPatternMatching("EURUSD=X", "1mo", "30m", csv_path="1_week_30min.csv")
    data_len = len(hist.data.index)
    history = hist.moving_frame_similarity(data_len - 100 - 1, data_len - 1) # columns=["Open"]
    hist.plot_similarity_history(history, ["Open", "High", "Low", "Close"])


test_moving_frame_similarity()

# hist = HistoricalPatternMatching("EURUSD=X", "1d", "30m", csv_path="data.csv")
# print(hist.fragments_similarity(0, 9, 10, 19))
# hist.data[["Open", "High", "Low", "Close"]].plot()
# plt.show()