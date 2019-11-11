import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
import random
from sklearn.metrics import roc_curve, roc_auc_score, auc
import sklearn.metrics as metrics
import pickle

sample = pickle.load(open("Embeddings/EMB_snn_4_25_16_full-75","rb"))
print(sample.head())