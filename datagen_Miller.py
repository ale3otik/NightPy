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


class TelegramCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        os.system("telegram-send 'Start NN training'")

    def on_train_end(self, logs={}):
        os.system("telegram-send 'End NN training'")

    def on_epoch_begin(self, epoch, logs={}):
        os.system("telegram-send 'Epoch {}'".format(epoch + 1))

    def on_epoch_end(self, epoch, logs={}):
        os.system("telegram-send 'loss: {:.4f}, val_loss: {:.4f}'".format(logs.get('loss'), logs.get('val_loss')))

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


class DataGenerator:
    def __init__(self, bad_columns=None, mean=None, sd=None):
        self.mean = mean
        self.sd = sd
        self.bad_columns = bad_columns
        
    def fit(self, X):
#         X = deepcopy(X).fillna(0)
        self.mean = X.mean()
        self.sd = X.std()
        print('end fit')
        
    def normalize(self, X):
        X  = (X - self.mean) / self.sd
        X = np.clip(X, a_min=-10, a_max=10)
        return X
    
    def do_fucking_job(self, X, y, feature_expander, 
                       features_win, forecast_win):
        
            assert min(features_win, forecast_win) > 0
            
            features_win -= 1
            forecast_win -= 1
            
            assert max(forecast_win, features_win) < X.shape[0]
            
            '''
                expander must return numpy array with shape (new_features_len)
            '''
            
            if feature_expander is not None:
                new_features = []
                for i in np.arange(features_win, X.shape[0]):
                    new_features.append(feature_expander(
                        X.iloc[i - features_win:i + 1]))
                new_features = np.array(new_features)
                wide = new_features.shape[1]
                dummy_cells = np.zeros((features_win, wide))
                new_features = np.concatenate((dummy_cells, new_features),
                                              axis=0)
                X = pd.concat((X, pd.DataFrame(new_features)),axis=1)
                
            if self.bad_columns is not None:
                X = X.drop(self.bad_columns, axis=1)
                
            Y = deepcopy(X)
            
            for i in np.arange(forecast_win) + 1:
                Y = pd.concat((Y, X.shift(periods=i)),axis=1)
            
            Y.columns = np.arange(len(Y.columns))
                
            Y = Y.fillna(Y.median(axis=0))
            return Y, y
            
    
    def flow(self, X, y, batch_size=10000, feature_expender=None, 
             forecast_win=10,features_win=100):
        ranges = np.arange(X.shape[0])
        assert batch_size <= len(ranges)
        while True:
            np.random.shuffle(ranges)
            for i in np.arange(0, len(ranges) - batch_size + 1, batch_size):
                inds = ranges[i:i + batch_size]
                x_batch, y_batch = X.iloc[inds], y[inds]
                x_batch, y_batch = self.do_fucking_job(x_batch, y_batch,
                                feature_expender, forecast_win, features_win)
                #  USE normalize if you REALLY need
                x_batch = self.normalize(x_batch)
                x_batch = np.array(x_batch)
                print('batch released')
                yield (x_batch, y_batch)     