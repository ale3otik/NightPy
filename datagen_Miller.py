import numpy as np 
import scipy.stats as sps
import pandas as pd
import h5py
import os
from tqdm import tqdm

from copy import deepcopy

from numpy import array
from os import environ
from os.path import join
from sys import argv

from glob import glob
from numpy import zeros
from os.path import basename, join

from keras.models import load_model
from os import environ
from os.path import abspath, dirname, join

from copy import copy

from sklearn.cross_validation import train_test_split


import keras
from keras import backend as K

from keras.models import Sequential
from keras.models import model_from_json

from keras.layers import InputLayer, Input, Dense, Activation, Dropout, BatchNormalization, Reshape
from keras.layers import Average, Conv2D, MaxPooling2D, Flatten
from keras.layers.merge import Concatenate
from keras.activations import relu, softmax, sigmoid
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.regularizers import l2

from keras.constraints import maxnorm

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

from keras.losses import binary_crossentropy, mse

from keras.callbacks import LearningRateScheduler


def score(loss):
    return 10 / (loss + 1e-9) ** 0.5

class TelegramCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        os.system("telegram-send 'Start NN training'")

    def on_train_end(self, logs={}):
        os.system("telegram-send 'End NN training'")

    def on_epoch_begin(self, epoch, logs={}):
        os.system("telegram-send 'Epoch {}'".format(epoch + 1))

    def on_epoch_end(self, epoch, logs={}):
        os.system("telegram-send 'score: {:.4f}, val_score: {:.4f}'".format(
            score(logs.get('loss')), score(logs.get('val_loss'))))

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


class DataGenerator:
    def __init__(self, batch_size=10000, feature_expander=None, 
             forecast_win=10,features_win=100, isLstm=False,
            bad_columns=None, mean=None, sd=None):

        assert min(features_win, forecast_win) > 0
        features_win -= 1
        forecast_win -= 1

        self.batch_size = batch_size
        self.feature_expander = feature_expander
        self.bad_columns = bad_columns
        self.forecast_win = forecast_win
        self.features_win = features_win
        self.isLstm = isLstm
        self.mean = mean
        self.sd = sd
        self.bad_columns = bad_columns
        

    def fit(self, X=None):
        if X is not None:
            X = deepcopy(X).fillna(0)
            self.mean = X.mean()
            self.sd = X.std()
            print('end fit')
        

    def normalize(self, X):
        if self.mean is not None:
            X  = (X - self.mean) / self.sd
            X = np.clip(X, a_min=-10, a_max=10)
        return X
    

    def do_fucking_job(self, X, y):

        assert max(self.forecast_win, self.features_win) < X.shape[0]
        
        if self.feature_expander is not None:
            new_features = []
            for i in np.arange(self.features_win, X.shape[0]):
                new_features.append(self.feature_expander(
                    X.iloc[i - self.features_win:i + 1]))
            new_features = np.array(new_features)
            wide = new_features.shape[1]
            dummy_cells = np.zeros((self.features_win, wide))
            new_features = np.concatenate((dummy_cells, new_features),
                                          axis=0)


            new_features = pd.DataFrame(new_features)
            X.index = new_features.index

            X = pd.concat((X, new_features),axis=1)

            
        if self.bad_columns is not None:
            X = X.drop(self.bad_columns, axis=1)
            
        X = deepcopy(X)
        
        X.columns = np.arange(len(X.columns))

            
        X = X.fillna(X.median(axis=0))

        # print(Y.shape[1])

        return X, y
            
    
    def flow(self, X, y):
        ranges = np.arange(X.shape[0])
        assert self.batch_size <= len(ranges)
        while True:
            #for i in np.arange(0, len(ranges) - self.batch_size + 1, self.batch_size):
                i = np.random.randint(len(ranges) - self.batch_size)
                inds = ranges[i:i + self.batch_size]
                x_batch, y_batch = X.iloc[inds], y[inds]
                x_batch_tea, x_batch_coffee = x_batch.iloc[:, :x_batch.shape[1] // 2], \
                                                x_batch.iloc[:, x_batch.shape[1] // 2:]
                x_batch_tea, y_batch = self.do_fucking_job(x_batch_tea, y_batch)
                x_batch_coffee, y_batch = self.do_fucking_job(x_batch_coffee, y_batch)
                #  USE normalize if you REALY need
                # x_batch = self.normalize(x_batch)
                x_batch_tea = np.array(x_batch_tea, dtype=np.float32)
                x_batch_coffee = np.array(x_batch_coffee, dtype=np.float32)
                x_batch = np.concatenate((x_batch_tea, x_batch_coffee), axis=1)
                y_batch = y_batch.reshape(-1, 1)
                if self.isLstm:
                    x_batch = [x_batch.reshape([1] + list(x_batch.shape)), np.zeros(1, 200), np.zeros(1, 200)]
                    y_batch = [y_batch.reshape(1, -1), np.zeros(1, 200), np.zeros(1, 200)]
                # print('batch released')
                yield (x_batch, y_batch)     


