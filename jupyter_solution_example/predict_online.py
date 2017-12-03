#!/usr/bin/python

from __future__ import print_function  # for python 2 compatibility
import hackathon_protocol
import os
import pandas as pd
from feature_generator import count_all_features
import numpy as np
from keras.models import load_model

USERNAME = "Andrei_potishe_pliz"
PASSWORD = "antoha322"

CONNECT_IP = os.environ.get("HACKATHON_CONNECT_IP") or "127.0.0.1"
CONNECT_PORT = int(os.environ.get("HACKATHON_CONNECT_PORT") or 12345)


class MyClient(hackathon_protocol.Client):
    def __init__(self, sock):
        super(MyClient, self).__init__(sock)
        print('on_init')
        self.counter = 0
        self.target_instrument = 'TEA'
        self.send_login(USERNAME, PASSWORD)
        self.last_raw = None

        self.buffer_main = pd.DataFrame()
        self.buffer_with_features = pd.DataFrame()

        # Load pre-trained model previously created by create_model.ipynb
        self.model = load_model('weights.04-0.04.hdf5')

        self.win_size = 10
        self.counter = 0

    def on_header(self, csv_header):
        print('onheader')
        self.header = {column_name: n for n, column_name in enumerate(csv_header)}
        self.columns = csv_header
        print(csv_header)
        # print("Header:", self.header)

    def on_orderbook(self, cvs_line_values):
        if cvs_line_values[0] == 'COFFEE':
            return

        print('onorderbook')
        

        self.buffer_main = self.buffer_main.append(
            pd.DataFrame(np.array([np.array(cvs_line_values)]), columns=self.columns), ignore_index=True)
        print(self.buffer_with_features.shape)
        print(self.buffer_main.shape)

        if self.buffer_main.shape[0] >= self.win_size:
            print(self.buffer_main.columns)
            print(self.buffer_main.iloc[0])
            features = count_all_features(self.buffer_main.iloc[-self.win_size:])
            to_append = pd.DataFrame(np.concatenate([np.array(cvs_line_values), features]))
            print('to_append', to_append.shape)

            self.buffer_with_features = self.buffer_with_features.append(to_append, ignore_index=True)
            if self.buffer_with_features.shape[0] > self.win_size:
                self.buffer_with_features = self.buffer_with_features.iloc[-self.win_size:]

        print('finish_onorder')

    def make_prediction(self):
        print('make_prediction')
        print('buffer shape', self.buffer_with_features.shape)
        prediction = self.model.predict([self.buffer_with_features])
        print('pred shape', prediction.shape)
        answer = prediction[0][-1]
        self.send_volatility(answer)

    def on_score(self, items_processed, time_elapsed, score_value):
        print('on_score')
        print("Completed! items processed: %d, time elapsed: %.3f sec, score: %.6f" % (
        items_processed, time_elapsed, score_value))
        self.stop()


def on_connected(sock):
    client = MyClient(sock)
    client.run()


def main():
    hackathon_protocol.tcp_connect(CONNECT_IP, CONNECT_PORT, on_connected)


if __name__ == '__main__':
    main()
