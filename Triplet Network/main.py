import tensorflow as tf
import pickle
import sys
import matplotlib.pyplot as plt1
import numpy as np
from IPython.display import Audio, display
import time
from sklearn.model_selection import train_test_split
import random
from network import *
from data import *
import os

'''
- loss = cosine, problem with euclidean
- net = Vanilla/Denset
- layers: number of layers
- neurons: number of neurons per layer
- embedding length: 8, 16, 32
- epoch: 50, 75, 100, 200
- (densenet param: subject to results)
- samples per perturbagen: 50, 100, 200

sys arguments: layer neuron emb_len dropout samples_per_pert 

moral: layers is less, neuron is more
'''

# command-line arguments
layer = int(sys.argv[1])
neuron = int(sys.argv[2])
emb_len = int(sys.argv[3])
dropout = float(sys.argv[4])
samples_per_pert = int(sys.argv[5])

print(layer, neuron, emb_len, dropout, samples_per_pert)
model_name = "MOD_triplet_" + str(layer) + "_" + str(neuron) + "_" + str(emb_len) \
             + "_" + str(dropout) + "_" + str(samples_per_pert)
embedding_name = "EMB_triplet_" + str(layer) + "_" + str(neuron) + "_" + str(emb_len) \
                 + "_" + str(dropout) + "_" + str(samples_per_pert)
if (os.path.exists('../Models/' + model_name) == 0):
    os.mkdir('../Models/' + model_name)

# Load necessary files
full = pickle.load(open('../Data/full', 'rb'))
dbfile = open('../Data/test_perts', 'rb')
test_pert = pickle.load(dbfile)
dbfile = open('../Data/train_perts', 'rb')
train_pert = pickle.load(dbfile)
dbfile.close()

# List of all perturbagens
all_pert = np.concatenate((train_pert, test_pert))

# Testing on unseen perturbagens
epoch = 30
s = siamese("cos", "net", layer, neuron, emb_len, dropout)
print("= Created Model")
split = 90
X, test = get_data(full, all_pert[0:30], samples_per_pert, split)
print(X.shape, test.shape)
print("== Got Data")

# create argument dictionary
input_list = ["model_name", "emb_name", "siamese", "epoch", "X_train", "X_test", "full", "all_pert"]
arguments_list = [model_name, embedding_name, s, epoch, X, test, full, all_pert]
input_dict = dict()
for i in range(len(input_list)):
    input_dict[input_list[i]] = arguments_list[i]

# Run network
results = run_network(input_dict)

print("===== Done")
