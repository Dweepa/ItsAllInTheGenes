import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import chain

import keras
from keras.datasets import reuters
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers.noise import AlphaDropout
from keras.preprocessing.text import Tokenizer
from keras.layers import Layer
from tensorflow.python.keras import backend as K

def create_base_network(n_dense=6,
                   dense_units=16,
                   activation='selu',
                   dropout=AlphaDropout,
                   dropout_rate=0.1,
                   kernel_initializer='lecun_normal',
                   optimizer='adam',
                   num_classes=1,
                   max_words=978):

    model = Sequential()
    model.add(Dense(dense_units, input_shape=(max_words,),
                    kernel_initializer=kernel_initializer))
    model.add(Activation(activation))
    model.add(dropout(dropout_rate))

    for i in range(n_dense - 1):
        model.add(Dense(dense_units, kernel_initializer=kernel_initializer))
        model.add(Activation(activation))
        model.add(dropout(dropout_rate))
    return model
