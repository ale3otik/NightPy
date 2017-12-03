def keras_lstm_model_2(max_sequence_length=None, input_shape=None, lstm_units=200, eps_reg=1e-4):
    input = Input(shape=(max_sequence_length, input_shape))
    norm = BatchNormalization()(input)
    lstm_layer = LSTM(units=lstm_units ,return_sequences=True)(norm)
    
    nnet = Dropout(0.1)(lstm_layer) 
    
    dense = Dense(50, input_shape=(max_sequence_length, lstm_units),
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=regularizers.l2(eps_reg),
                    activation=linear)(nnet)

    output = Dense(1, input_shape=(max_sequence_length, lstm_units),
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=regularizers.l2(eps_reg),
                    activation=linear)(dense)
    # print(lstm_layer.shape)
    # print(dense.shape)
    # print(input.shape)
    model = Model(input, output)
    optimizer = optimizers.Adam(1e-3, clipvalue=100.0, decay=1e-3)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[score])
    return model