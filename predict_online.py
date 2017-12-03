#!/usr/bin/python

from __future__ import print_function  # for python 2 compatibility
import hackathon_protocol
import os
import pandas as pd
from feature_generator import count_all_features
import numpy as np
from keras.models import load_model
import serg_model

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
        self.last_tea = None
        self.last_cofee = None
        self.buffer_main_tea = pd.DataFrame()
        self.buffer_with_features_tea = pd.DataFrame()
        self.buffer_main_coffee = pd.DataFrame()
        self.buffer_with_features_coffee = pd.DataFrame()
        
        self.h = np.array([np.zeros(200,dtype=np.float32)])
        self.c = np.array([np.zeros(200,dtype=np.float32)])

        # Load pre-trained model previously created by create_model.ipynb
        self.model = serg_model.reccurent_model(input_shape=80)
        # self.model = keras_ls   .reccurent_model(input_shape=262)
        self.model.load_weights('weights.013-0.029.hdf5')

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
            self.last_cofee = np.array(cvs_line_values[2:],dtype=np.float32)
            # cvs_line_values = cvs_line_values[2:]
            # self.buffer_main_coffee = self.buffer_main_coffee.append(
            #     pd.DataFrame(np.array([np.array(cvs_line_values)]), columns=self.columns), ignore_index=True)

            # if self.buffer_main_coffee.shape[0] >= self.win_size:
            #     # features = count_all_features(self.buffer_main_coffee.iloc[-self.win_size:])
            #     to_append = pd.DataFrame(np.concatenate([np.array(cvs_line_values), np.array([])])).T

            #     self.buffer_with_features_coffee = self.buffer_with_features_coffee.append(to_append, ignore_index=True)
            #     if self.buffer_with_features_coffee.shape[0] > self.win_size:
            #         self.buffer_with_features_coffee = self.buffer_with_features_coffee.iloc[-self.win_size:]

        
        if cvs_line_values[0] == 'TEA':
            self.last_tea = np.array(cvs_line_values[2:],dtype=np.float32)
            # cvs_line_values = cvs_line_values[2:]
            # self.buffer_main_tea = self.buffer_main_tea.append(
            #     pd.DataFrame(np.array([np.array(cvs_line_values)]), columns=self.columns), ignore_index=True)

            # # if self.buffer_main_tea.shape[0] >= self.win_size:
            # #     features = count_all_features(self.buffer_main_tea.iloc[-self.win_size:])
            # #     to_append = pd.DataFrame(np.concatenate([np.array(cvs_line_values), features])).T
            # if self.buffer_main_tea.shape[0] >= self.win_size:
            #     # features = count_all_features(self.buffer_main_tea.iloc[-self.win_size:])
            #     to_append = pd.DataFrame(np.concatenate([np.array(cvs_line_values), np.array([])])).T

            #     self.buffer_with_features_tea = self.buffer_with_features_tea.append(to_append, ignore_index=True)
            #     if self.buffer_with_features_tea.shape[0] > self.win_size:
            #         self.buffer_with_features_tea = self.buffer_with_features_tea.iloc[-self.win_size:]

    def make_prediction(self):
        # print('make_prediction')
        # print('buffer shape', self.buffer_with_features_tea.shape)
        # input = np.concatenate([self.buffer_with_features_tea, self.buffer_with_features_coffee], axis=-1)
        input = np.concatenate([self.last_tea,self.last_cofee])
        input = input.reshape((1,1,input.shape[0]))
        # print(input.shape)
        # print(self.state1.shape)
        # print(input.shape)
        prediction, self.h, self.c = self.model.predict([input,self.h, self.c])

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
