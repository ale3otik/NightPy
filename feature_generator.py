import numpy as np
import pandas as pd
import sys

def general_features(df):
    row = df.iloc[-1]
    features = [row['ASK_V_' + str(i)] - row['BID_V_' + str(i)] for i in range(1, 11)] + \
    [row['ASK_P_' + str(i)] - row['BID_P_' + str(i)] for i in range(1, 11)] + \
    [row['ASK_V_' + str(i)] * row['ASK_P_' + str(i)] - row['BID_V_' + str(i)] * row['BID_P_' + str(i)] for i in range(1, 11)] + \
    [np.sum([row['ASK_V_' + str(i)] for i in range(1,11)])] + \
    [np.sum([row['ASK_V_' + str(i)] * row['ASK_P_' + str(i)] for i in range(1,11)])] + \
    [np.sum([row['BID_V_' + str(i)] for i in range(1,11)])] + \
    [np.sum([row['BID_V_' + str(i)] * row['BID_P_' + str(i)] for i in range(1,11)])] + \
    [row['ASK_V_' + str(i)] / row['ASK_V_' + str(i - 1)] for i in range(2, 11)] + \
    [row['ASK_P_' + str(i)] / row['ASK_P_' + str(i - 1)] for i in range(2, 11)] + \
    [row['BID_V_' + str(i)] / row['BID_V_' + str(i - 1)] for i in range(2, 11)] + \
    [row['BID_P_' + str(i)] / row['BID_P_' + str(i - 1)] for i in range(2, 11)] + \
    [row['BID_V_' + str(i)] * row['BID_P_' + str(i)] / (row['BID_V_' + str(i - 1)] * row['BID_P_' + str(i - 1)]) for i in range(2, 11)] + \
    [row['ASK_V_' + str(i)] * row['ASK_P_' + str(i)] / (row['ASK_V_' + str(i - 1)] * row['ASK_P_' + str(i - 1)]) for i in range(2, 11)]
    return features

def spread_tightness(df):
    return (df.iloc[-1]['ASK_P_1'] - df.iloc[-1]['BID_P_1']) / df.iloc[-1]['BID_P_1']

def valueWP(df, V):
    row = df.iloc[-1]
    sum_V = 0
    sum_P = 0
    for i in range(1, 11):
        pv = row['BID_P_' + str(i)] * row['BID_V_' + str(i)]
        if sum_V > V:
            sum_P += 0
        elif sum_V + pv > V:
            sum_P += (V - sum_V) * row['BID_P_' + str(i)]
        else:
            sum_P += pv * row['BID_P_' + str(i)]
        row['BID_P_' + str(i)] * row['BID_V_' + str(i)]
        sum_V += pv
    return sum_P

def assymetry(df):
    row = df.iloc[-1]
    n_ask = np.sum([row['ASK_V_' + str(i)] for i in range(1, 11)])
    n_bid = np.sum([row['BID_V_' + str(i)] for i in range(1, 11)])
    return abs(n_ask / (n_ask + n_bid) - 0.5) / 0.5

def OIR_VOI(df): #Order Imbalanced Ratio
    row = df.iloc[-1]
    prev = df.iloc[-2]
    volume_bid_orders = np.sum([row['BID_V_' + str(i)] for i in range(1,11)])
    volume_ask_orders = np.sum([row['ASK_V_' + str(i)] for i in range(1,11)])
    volume_bid_volume = np.sum([row['BID_V_' + str(i)] * row['BID_P_' + str(i)] for i in range(1,11)])
    volume_ask_volume = np.sum([row['ASK_V_' + str(i)] * row['ASK_P_' + str(i)] for i in range(1,11)])
    price_bid = row['BID_P_1'] #best bid price
    price_ask = row['ASK_P_1'] #best ask price
    
    volume_bid_orders_prev = np.sum([prev['BID_V_' + str(i)] for i in range(1,11)])
    volume_ask_orders_prev = np.sum([prev['ASK_V_' + str(i)] for i in range(1,11)])
    volume_bid_volume_prev = np.sum([prev['BID_V_' + str(i)] * prev['BID_P_' + str(i)] for i in range(1,11)])
    volume_ask_volume_prev = np.sum([prev['ASK_V_' + str(i)] * prev['ASK_P_' + str(i)] for i in range(1,11)])
    price_bid_prev = prev['BID_P_1'] #best bid price
    price_ask_prev = prev['ASK_P_1'] #best ask price
    
    ratio_orders = (volume_bid_orders - volume_ask_orders) / (volume_bid_orders + volume_ask_orders)
    ratio_volume = (volume_bid_volume - volume_ask_volume) / (volume_bid_volume + volume_ask_volume)
    
    if price_bid < price_bid_prev:
        Vbid_orders = 0
        Vbid_volume = 0
    elif price_bid == price_bid_prev:
        Vbid_orders = volume_bid_orders - volume_bid_orders_prev
        Vbid_volume = volume_bid_volume - volume_bid_volume_prev
    else:
        Vbid_orders = volume_bid_orders
        Vbid_volume = volume_bid_volume

    if price_ask < price_ask_prev:
        Vask_orders = volume_ask_orders
        Vask_volume = volume_ask_volume
    elif price_ask == price_ask_prev:
        Vask_orders = volume_ask_orders - volume_ask_orders_prev
        Vask_volume = volume_ask_volume - volume_ask_volume_prev
    else:
        Vask_orders = 0
        Vask_volume = 0
    
    return [ratio_orders, ratio_volume, Vbid_orders - Vask_orders, Vbid_volume - Vbid_volume]

def new_features(df): #накопленный объем заявок для k реек для бида и аска
    features = []
    row = df.iloc[-1]

    cum_volume_bid_orders = [np.sum([row['BID_V_' + str(i)] for i in range(1, j)]) for j in range(2, 11)]
    cum_volume_ask_orders = [np.sum([row['ASK_V_' + str(i)] for i in range(1, j)]) for j in range(2, 11)]

    cum_volume_bid_volume = [np.sum([row['BID_V_' + str(i)] * row['BID_P_' + str(i)] for i in range(1, j)]) for j in range(2, 11)]
    cum_volume_ask_volume = [np.sum([row['ASK_V_' + str(i)] * row['ASK_P_' + str(i)] for i in range(1, j)]) for j in range(2, 11)]

    features += [cum_volume_bid_orders, cum_volume_ask_orders, cum_volume_bid_volume, cum_volume_ask_volume]

    bid_to_ask_orders = [cum_volume_bid_orders[i] / cum_volume_ask_orders[i] for i in range(0, 9)]
    bid_to_ask_volume = [cum_volume_bid_volume[i] / cum_volume_ask_volume[i] for i in range(0, 9)]

    features += [bid_to_ask_orders, bid_to_ask_volume]

    bid_av_w_price = [cum_volume_bid_volume[i] / cum_volume_bid_orders[i] for i in range(0, 9)]
    ask_av_w_price = [cum_volume_ask_volume[i] / cum_volume_ask_orders[i] for i in range(0, 9)]

    features += [bid_av_w_price, ask_av_w_price]

    bid_to_ask = [bid_av_w_price[i] / ask_av_w_price[i] for i in range(0, 9)]

    features += [bid_to_ask,]

    window = 10

    row = df.tail(window)

    bid_volume_window_order = []
    ask_volume_window_order = []

    bid_volume_window_volume = []
    ask_volume_window_volume = []

    for j in range(0, window):

        bid_volume_window_order.append(np.sum([row['BID_V_' + str(i)].iloc[j] for i in range(1, 11)]))
        ask_volume_window_order.append(np.sum([row['ASK_V_' + str(i)].iloc[j] for i in range(1, 11)]))

        bid_volume_window_volume.append(np.sum([row['BID_V_' + str(i)].iloc[j] * row['BID_P_' + str(i)].iloc[j] for i in range(1, 11)]))
        ask_volume_window_volume.append(np.sum([row['ASK_V_' + str(i)].iloc[j] * row['ASK_P_' + str(i)].iloc[j] for i in range(1, 11)]))

    av_bid_volume_window_order = np.sum(bid_volume_window_order) / window
    av_ask_volume_window_order = np.sum(ask_volume_window_order) / window

    av_bid_volume_window_volume = np.sum(bid_volume_window_volume) / window
    av_ask_volume_window_volume = np.sum(ask_volume_window_volume) / window

    max_bid_volume_window_order = np.max(bid_volume_window_order)
    max_ask_volume_window_order = np.max(ask_volume_window_order)

    max_bid_volume_window_volume = np.max(bid_volume_window_volume) / window
    max_ask_volume_window_volume = np.max(ask_volume_window_volume) / window

    min_bid_volume_window_order = np.min(bid_volume_window_order)
    min_ask_volume_window_order = np.min(ask_volume_window_order)

    min_bid_volume_window_volume = np.min(bid_volume_window_volume) / window
    min_ask_volume_window_volume = np.min(ask_volume_window_volume) / window
    
    result = []
    for i in features:
        if type(i) == list:
            for j in i:
                result.append(j)
        else:
            result.append(i)
    return result

def count_all_features(window):
    mean_volume = np.mean([window['BID_V_' + str(i)] * window['BID_P_' + str(i)] for i in range(1, 11)])
    features = np.array([spread_tightness(window)] + 
                        [valueWP(window, mean_volume)] + 
                        [assymetry(window)] +
                        general_features(window) + 
                        new_features(window))
    return features