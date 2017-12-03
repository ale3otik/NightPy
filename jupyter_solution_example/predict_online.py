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
        # print('on_init')
        self.counter = 0
        self.target_instrument = 'TEA'
        self.send_login(USERNAME, PASSWORD)
        self.last_raw = None

        self.buffer_main_tea = pd.DataFrame()
        self.buffer_with_features_tea = pd.DataFrame()
        self.buffer_main_coffee = pd.DataFrame()
        self.buffer_with_features_coffee = pd.DataFrame()

        # Load pre-trained model previously created by create_model.ipynb
        self.model = load_model('weights.08-0.03.hdf5')
        self.model.get_layer(name='lstm_10').stateful=True
        print(self.model.summary())

        self.win_size = 50
        self.counter = 0

    def on_header(self, csv_header):
        # print('onheader')
        self.header = {column_name: n for n, column_name in enumerate(csv_header)}
        self.columns = csv_header[2:]
        # print(self.columns)
        # print(len(self.columns))
        # print(csv_header)
        # print("Header:", self.header)

    def on_orderbook(self, cvs_line_values):
        if cvs_line_values[0] == 'COFFEE':
            cvs_line_values = cvs_line_values[2:]
            self.buffer_main_coffee = self.buffer_main_coffee.append(
                pd.DataFrame(np.array([np.array(cvs_line_values)]), columns=self.columns), ignore_index=True)

            if self.buffer_main_coffee.shape[0] >= self.win_size:
                features = count_all_features(self.buffer_main_coffee.iloc[-self.win_size:])
                to_append = pd.DataFrame(np.concatenate([np.array(cvs_line_values), features])).T

                self.buffer_with_features_coffee = self.buffer_with_features_coffee.append(to_append, ignore_index=True)
                if self.buffer_with_features_coffee.shape[0] > self.win_size:
                    self.buffer_with_features_coffee = self.buffer_with_features_coffee.iloc[-self.win_size:]
        
        if cvs_line_values[0] == 'TEA':
            cvs_line_values = cvs_line_values[2:]
            self.buffer_main_tea = self.buffer_main_tea.append(
                pd.DataFrame(np.array([np.array(cvs_line_values)]), columns=self.columns), ignore_index=True)

            # if self.buffer_main_tea.shape[0] >= self.win_size:
            #     features = count_all_features(self.buffer_main_tea.iloc[-self.win_size:])
            #     to_append = pd.DataFrame(np.concatenate([np.array(cvs_line_values), features])).T
        if self.buffer_main.shape[0] >= self.win_size:
            print(self.buffer_main.columns)
            print(self.buffer_main.iloc[0])
            features = count_all_features(self.buffer_main.iloc[-self.win_size:])
            to_append = pd.DataFrame(np.concatenate([np.array(cvs_line_values), features])).T
            print('to_append', to_append.shape)

                self.buffer_with_features_tea = self.buffer_with_features_tea.append(to_append, ignore_index=True)
                if self.buffer_with_features_tea.shape[0] > self.win_size:
                    self.buffer_with_features_tea = self.buffer_with_features_tea.iloc[-self.win_size:]


    def make_prediction(self):
        # print('make_prediction')
        # print('buffer shape', self.buffer_with_features_tea.shape)
        input = np.concatenate([self.buffer_with_features_tea, self.buffer_with_features_coffee],axis=-1)
        input = input.reshape([1] + list(input.shape))
        # print(input.shape)
        prediction = self.model.predict(input[:,-1:,:])
        # print('pred shape', prediction.shape)
        answer = prediction[0,:,0][-1]
        self.send_volatility(float(answer))

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
