import keras.backend as K
from keras.layers import Dense, Activation, Input, LSTM, Dropout, multiply, Lambda
from keras.layers import RNN
from keras.models import Model, Sequential
from keras import optimizers
from keras import regularizers 
import tensorflow as tf 
from keras.activations import sigmoid, linear
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

# class PhasedLSTMCell(keras.layers.Layer):

#     def __init__(self, units, cell,  **kwargs):
#         self.units = units
#         self.state_size = units
#         self.cell = cell
#         super(PhasedLSTMCell, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.built = True

#     def call(self, inputs, states):
#         tuple_inputs = tf.split(inputs, (inputs.shape[-1]-1,1), axis=-1)
#         return self.cell(tuple_inputs, states)

# class MyLayer(Layer):
#   def __init__(self, lstm_units=50,max_sequence_length=100):
#     self.lstm_units = lstm_units
#     self.max_sequence_length=max_sequence_length
#     self.cell = tf.contrib.rnn.PhasedLSTMCell(num_units=self.lstm_units)
#     super(MyLayer, self).__init__()

#   def build(self, input_shape):
#     self.shape = input_shape
#     super(MyLayer, self).build(input_shape)

#   def call(self, inputs, **kwargs):
#     features, times = tf.split(inputs,[inputs.get_shape().as_list()[-1]-1, 1] ,axis=-1)
#     print('feature', features.shape)
#     print('times' , times.shape)
#     lstm_layer, state = tf.nn.dynamic_rnn(self.cell, inputs=(features, times), dtype=tf.float32)
#     print('lstm', lstm_layer.shape)
#     return lstm_layer

#   def compute_output_shape(self, input_shape):
#     return (None, self.max_sequence_length, self.cell.output_size)

# def keras_phased_lstm_model(max_sequence_length=None, input_shape=None, lstm_units=50):
#     input = Input(shape=(max_sequence_length, input_shape))
#     timeseries = Input(shape=(max_sequence_length, 1))
#     # timestamps = Input(shape=(max_sequence_length, 1))
#     # cell = tf.contrib.rnn.LSTMCell(num_units=lstm_units)
#     print(input)
#     # cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_units, activation=tf.tanh)
#     # lstm_layer, state = tf.nn.dynamic_rnn(cell, inputs=input,dtype=tf.float32)
#     lstm_layer = MyLayer(50,max_sequence_length)(tf.concat([input, timeseries], axis=-1))
#     dense = Dense(1, input_shape=(max_sequence_length, lstm_units))(lstm_layer)
#     print(dense.shape)
#     model = Model([input,timeseries], dense)
#     optimizer = optimizers.Adam()
#     model.compile(loss='mean_squared_error', optimizer=optimizer)
#     return model

def keras_lstm_model_1(max_sequence_length=None, input_shape=None, lstm_units=100, eps_reg=1e-2):
    input = Input(shape=(max_sequence_length, input_shape))
    lstm_layer = LSTM(units=lstm_units ,return_sequences=True)(input)
    
    dense = Dense(1, input_shape=(max_sequence_length, lstm_units),
                    kernel_initializer='normal',
                    kernel_regularizer=regularizers.l2(eps_reg),
                    activation=linear)(lstm_layer)

    # print(lstm_layer.shape)
    # print(dense.shape)
    # print(input.shape)
    model = Model(input, dense)
    optimizer = optimizers.Adam()
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
