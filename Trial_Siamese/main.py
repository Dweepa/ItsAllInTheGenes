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


def done():
    framerate = 44100
    duration = 0.6
    freq = 300
    t = np.linspace(0, duration, framerate * duration)
    data = np.sin(2 * np.pi * freq * t)
    display(Audio(data, rate=framerate, autoplay=True))


X = pickle.load(open('../Data/X_train_triplet_full', 'rb'))
test = pickle.load(open('../Data/X_test_triplet_full', 'rb'))
y = pickle.load(open('../Data/y_test', 'rb'))

# Testing on unseen perturbagens
epochs = 20
s = siamese("euc", "net")
embeddings, trained, pred, p_loss, n_loss, train_acc_l, test_acc_l = run_network(s, epochs, X, test)

print("Testing on unseen perturbagens: Euclidean distance")
p = np.sum(trained[0][2] <= 0.5)
n = np.sum(trained[0][3] > 0.5)
print("Training Accuracy", (p + n) / len(X[0]) / 2)

p = np.sum(pred[0][2] <= 0.5)
n = np.sum(pred[0][3] > 0.5)
print("Testing Accuracy", (p + n) / len(test[0]) / 2)

# Cosine on unseen perturbagens
s = siamese("cos", "net")
embeddings, trained, pred, p_loss, n_loss, train_acc_l, test_acc_l = run_network(s, epochs, X, test)

print("Testing on unseen perturbagens: Cosine distance")
p = np.sum(trained[0][2] <= 0.5)
n = np.sum(trained[0][3] > 0.5)
print("Training Accuracy", (p + n) / len(X[0]) / 2)

p = np.sum(pred[0][2] <= 0.5)
n = np.sum(pred[0][3] > 0.5)
print("Testing Accuracy", (p + n) / len(test[0]) / 2)

# Testing on seen perturbagens
