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
data.head()

# Obtain targets
target = get_target_labels(data, metadata)
print(len(target))

# Attach labels
data1 = data.transpose()
data1['target'] = target
data = data1.sort_values('target')
data.head()

# Create the 2 dictionaries
location_pert = create_location_pert(data)

pert2profiles = create_pert2profile(data)

train, test = train_and_test_perturbagens(np.unique(data.target))

X_train, y_train = generate_data(data, train, 2)
X_test, y_test = generate_data(data, test, 2)

dbfile = open('X_train', 'ab')
pickle.dump(X_train, dbfile)
dbfile.close()

dbfile = open('y_train', 'ab')
pickle.dump(y_train, dbfile)
dbfile.close()

dbfile = open('X_test', 'ab')
pickle.dump(X_test, dbfile)
dbfile.close()

dbfile = open('y_test', 'ab')
pickle.dump(y_test, dbfile)
dbfile.close()

dbfile = open('X_train', 'rb')
db = pickle.load(dbfile)
