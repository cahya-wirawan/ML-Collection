from keras.models import Model
from keras.layers import Dropout
from keras.layers import (
    Dense,
    Input,
    Activation,
    Dense,
    Flatten,
    Embedding,
    LSTM,
    CuDNNGRU,
    GRU,
    Bidirectional
)
from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


# define cnn_lstm model
def cnn_lstm(dropout=0.1, activation='relu',
             max_seq_length=100, output_dim=100, embedding_matrix=None):

    input = Input(shape=(max_seq_length,), name='input')
    block = Embedding(len(embedding_matrix), output_dim, weights=[embedding_matrix],
                      input_length=max_seq_length, trainable=False)(input)
    block = Conv1D(filters=32, kernel_size=3, padding='same', activation=activation)(block)
    block = MaxPooling1D(pool_size=2)(block)
    block = LSTM(100)(block)
    block = Dense(12, activation=activation)(block)
    output = Dense(6, activation='sigmoid')(block)

    model = Model(inputs=[input], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# define gru model
def gru(dropout=0.1, activation='relu',
        max_seq_length=100, output_dim=100, embedding_matrix=None):

    input = Input(shape=(max_seq_length,), name='input')
    block = Embedding(len(embedding_matrix), output_dim, weights=[embedding_matrix],
                      input_length=max_seq_length, trainable=False)(input)
    gru_layer = CuDNNGRU(units=512)
    block = Bidirectional(gru_layer)(block)
    block = Dense(12, activation=activation)(block)
    block = Dropout(dropout)(block)
    output = Dense(6, activation='sigmoid')(block)

    model = Model(inputs=[input], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model