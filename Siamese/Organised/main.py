# Import necessary libraries

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers.noise import AlphaDropout
from keras.layers import Layer
from tensorflow.python.keras import backend as K

# Import custom modules
from network import *
from data import *

# Create the base network
network = dict(n_dense=10, dense_units=16, activation='selu', dropout=AlphaDropout, dropout_rate=0.1,
               kernel_initializer='lecun_normal', optimizer='sgd', num_classes=2)

shared_model = create_base_network(**network)

print("Shared model summary")
shared_model.summary()

# Create the siamese network
model = create_siamese_network(shared_model)
print("Siamese network model summary")
model.summary()

# Obtain data
gctxfile = "../../Data/Sig Annotated Level 5 Data.gctx"
jsonfile = "../../Data/sig-pert mapping.json"

data, metadata = gctx2pd(gctxfile, jsonfile)
print(data.head())

target = get_target_labels(data, metadata)
print(len(target))

data1 = data.transpose()
data1['target'] = get_target_labels(data1, metadata)
data = data1.sort_values('target')
print(data.head())

location_pert = create_location_pert(data)
pert2profile = create_pert2profile(data)

X, y = get_training_data(data, pert2profiles, location_pert, 10)

print(X.shape)

# Fit data
model.fit([X[0], X[1]], y, epochs=10, verbose=1)
