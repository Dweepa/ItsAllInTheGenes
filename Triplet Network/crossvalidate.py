# Import necessary libraries
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers.noise import AlphaDropout
from keras.layers import Layer
from tensorflow.python.keras import backend as K
from sklearn.model_selection import train_test_split
import pickle
from sklearn.model_selection import KFold

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

# List of all perturbagens
all_pert = np.concatenate((train_pert, test_pert))

from sklearn.model_selection import KFold

# cross validation code WIP
all_pert = np.concatenate((train_pert, test_pert))

'''
 TO-DO: make all variables system arguments: 
 - epochs
 - samples pert pert
 - architecture of siamese (layers and nodes per layer)
 - name of saved model file
 '''


def cross_validate(splitsize):
    kf = KFold(n_splits=splitsize)
    results = []
    c = 1
    for train_idx, val_idx in kf.split(all_pert):
        print("\nCrossvalidation run #", c)
        c += 1
        train_perts = all_pert[train_idx]
        val_perts = all_pert[val_idx]
        print(train_perts.shape, val_perts.shape)
        X_train = generate_data(data, train_perts, 100)
        print("\n")
        X_val = generate_data(data, val_perts, 100)
        s = siamese("cos", "net")
        epochs = 10
        print("\n")
        embeddings, trained, pred, p_loss, n_loss, train_acc_l, test_acc_l = run_network(s, epochs, X_train, X_val)

        p = np.sum(trained[0][2] <= 0.5)
        n = np.sum(trained[0][3] > 0.5)
        train_acc = ((p + n) / len(X_train[0]) / 2)

        p = np.sum(pred[0][2] <= 0.5)
        n = np.sum(pred[0][3] > 0.5)
        val_acc = ((p + n) / len(X_val[0]) / 2)

        accuracy = (train_acc, val_acc)
        results.append(accuracy)
    return results


result = cross_validate(5)
test_res = 0
train_res = 0
for i in result:
    test_res += i[1]
    train_res += i[0]
print("\n\nCross-validation result: Train: \n%s\t\t Test:%s" % (train_res, test_res))
