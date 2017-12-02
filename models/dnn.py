import keras.backend as K
from keras.layers import Dense, Activation, Input, LSTM, Dropout, multiply, Lambda
from keras.layers import RNN
from keras.models import Model, Sequential
from keras import optimizers
from keras import regularizers 
import tensorflow as tf 
from keras.activations import sigmoid, linear, tanh
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


def nn_1(input_shape=None, drop_val=0.1, eps_reg=1e-2, hidden_units=200):
    model = Sequential()
    model.add(Dense(hidden_units=hidden_units, input_shape=(input_shape,),
                    kernel_initializer='normal',
                    kernel_regularizer=regularizers.l2(0.01)
                    activation=tanh,
                   ))

    model.add(Dropout(drop_val))
    # model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(units=hidden_units,
                    kernel_initializer='normal',
                    kernel_regularizer=regularizers.l2(eps_reg)
                   ))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    
    model.add(Dense(units=1,
                    kernel_initializer='normal',
                    kernel_regularizer=regularizers.l2(eps_reg),
                    activation=linear,
                   ))
    optimizer = optimizers.Adam(optimizers.Adam(1e-3, clipvalue=10.0))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


