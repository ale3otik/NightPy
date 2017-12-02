import keras.backend as K
from keras.layers import Dense, Activation, Input, LSTM, Dropout, multiply, Lambda
from keras.layers import RNN
from keras.models import Model, Sequential
from keras import optimizers
def keras_lstm_model_1(max_sequence_length=None, input_shape=None, lstm_units=100):
    input = Input(shape=(max_sequence_length, input_shape))
    lstm_layer = LSTM(units=lstm_units ,return_sequences=True)(input)
    dense = Dense(1, input_shape=(max_sequence_length, lstm_units))(lstm_layer)
    # print(lstm_layer.shape)
    # print(dense.shape)
    # print(input.shape)
    model = Model(input, dense)
    optimizer = optimizers.Adam()
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
