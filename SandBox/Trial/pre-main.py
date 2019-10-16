# Import necessary libraries
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers.noise import AlphaDropout
from keras.layers import Layer
from tensorflow.python.keras import backend as K
import pickle

# Import custom modules
from network import *
from data import *

dbfile = open('../../Data/full', 'rb')
data = pickle.load(dbfile)
dbfile = open('../../Data/location_pert', 'rb')
location_pert = pickle.load(dbfile)
dbfile = open('../../Data/pert2profiles', 'rb')
pert2profiles = pickle.load(dbfile)
dbfile = open('../../Data/test_perts', 'rb')
test_pert = pickle.load(dbfile)
dbfile = open('../../Data/train_perts', 'rb')
train_pert = pickle.load(dbfile)

X_train = generate_data(data, train, 50)
X_test = generate_data(data, test, 50)

dbfile = open('X_train_triplet_full', 'ab')
pickle.dump(X_train, dbfile)
dbfile.close()

dbfile = open('X_test_triplet_full', 'ab')
pickle.dump(X_test, dbfile)
dbfile.close()
