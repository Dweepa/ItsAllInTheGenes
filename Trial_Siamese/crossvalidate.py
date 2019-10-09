# Import necessary libraries
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers.noise import AlphaDropout
from keras.layers import Layer
from tensorflow.python.keras import backend as K
from sklearn.model_selection import train_test_split
import pickle

# Import custom modules
from network import *
from data import *

dbfile = open('../Data/full', 'rb')
data = pickle.load(dbfile)
dbfile = open('../Data/location_pert', 'rb')
location_pert = pickle.load(dbfile)
dbfile = open('../Data/pert2profiles', 'rb')
pert2profiles = pickle.load(dbfile)
dbfile = open('../Data/test_perts', 'rb')
test_pert = pickle.load(dbfile)
dbfile = open('../Data/train_perts', 'rb')
train_pert = pickle.load(dbfile)
dbfile.close()

# X_train = generate_data(data,train_pert,100)
# X_test = generate_data(data,test_pert,100)

# dbfile = open('X_train_triplet_full', 'ab')
# pickle.dump(X_train, dbfile)
# dbfile.close()


X = pickle.load(open('../Data/X_train_triplet_full', 'rb'))
test = pickle.load(open('../Data/X_test_triplet_full', 'rb'))
# y = pickle.load(open('../Data/y_test', 'rb'))

# cross validation code WIP
all_pert = np.concatenate((train_pert, test_pert))

from sklearn.model_selection import KFold


def cross_validate(splitsize):
    kf = KFold(n_splits=splitsize)
    results = []
    for train_idx, val_idx in kf.split(all_pert):
        train_perts = all_pert[train_idx]
        val_perts = all_pert[val_idx]
        print(train_perts.shape, val_perts.shape)
        X_train = generate_data(data, train_perts, 42)
        X_val = generate_data(data, val_perts, 42)
        s = siamese("cos", "net")
        embeddings, trained, pred, p_loss, n_loss, train_acc_l, test_acc_l = run_network(s, epochs, X_train, X_val)

        p = np.sum(trained[0][2] <= 0.5)
        n = np.sum(trained[0][3] > 0.5)
        train_acc = ((p + n) / len(X[0]) / 2)

        p = np.sum(pred[0][2] <= 0.5)
        n = np.sum(pred[0][3] > 0.5)
        val_acc = ((p + n) / len(test[0]) / 2)

        accuracy = (train_acc, val_acc)
        results.append(accuracy)
    return results


result = cross_validate(5)
print("Cross-validation result: %s" % result)
# tf.summary.FileWriter('./logs', session.graph)
# saver = tf.train.Saver()
# saver.save(session, './models/trial2')
# K.clear_session()
