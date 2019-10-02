import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import random
import sys

embeddingname = sys.argv[1]

embeddings = pickle.load(open('./embeddings/'+embeddingname, 'rb'))

X = np.asarray(embeddings_small.loc[:, 'e1':'e32'])
y = np.asarray(embeddings_small['target'])

def softmax_function(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def label_positive(row, label, y):
    if y[int(row['index'])]==label:
        return True
    return False

def full_internal_evaluation(query_embedding, query_class, X, y, printinfo=False):
    cosines = np.dot(X, query_embedding.reshape(32, 1))
    softmax = softmax_function(cosines).flatten()
    sorted_softmax = list(zip(softmax, range(len(y))))
    sorted_softmax.sort(reverse=True)
    ordered = pd.DataFrame(sorted_softmax, columns=['softmax', 'index'])
    ordered['label'] = ordered.apply(lambda row: label_positive(row, query_class, y), axis=1)
    positives = ordered.index[ordered['label']==True].tolist()
    quantiles = []
    pos = 0
    total_negs = len(y)-len(positives)
    for a in range(len(positives)):
        quantiles.append((positives[a]-pos)/total_negs)
        pos+=1
        
    x_quantile = {}
    for a in np.arange(0, 1.001, 0.001):
        x_quantile[round(a, 3)] = 0
    # y_recall = []
    for a in quantiles:
        x_quantile[round(a, 3)]+=(1/len(quantiles))
    for a in np.arange(0.001, 1.001, 0.001):
        x_quantile[round(a, 3)]+=x_quantile[round(a-0.001, 3)]
    x_values = x_quantile.keys()
    y_values = x_quantile.values()

    imp_q = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    imp_q_val = [x_quantile[round(a, 3)] for a in imp_q]
    
    median = np.median(quantiles)
    if printinfo:
        print("Median Quantile:", median)

    results = pd.DataFrame(list(zip(imp_q, imp_q_val)), columns=['Quantile', 'Recall'])
    if printinfo:
        display(HTML(results.to_html()))
        plt.figure(figsize=[20,6])
        plt.xlabel("Quantile")
        plt.ylabel("Percentage of positive perturbagens below quantile")
        plt.plot(x_values, y_values)
        plt.show()
    auc = roc_auc_score(ordered['label'], ordered['softmax'])
    if printinfo:
#         plt.figure(figsize=[20,6])
#         fpr, tpr, thresholds = roc_curve(ordered['label'], ordered['softmax'])
#         plt.plot(fpr, tpr)
#         plt.show()
        
        plt.figure(figsize=[20,6])
        l = 100#positives[-10]
        colors = {True:'green', False:'red'}
        plt.scatter([a for a in range(l)], [0 for a in range(l)], c=ordered['label'][:l].apply(lambda x: colors[x]) )
        plt.show()

        plt.figure(figsize=[20,6])
        l = 100#positives[-10]
        plt.scatter([a for a in range(l)], [0 for a in range(l)], c=[a[1] for a in sorted_softmax[:l]] )
        plt.show()

        print("AUC: ", auc)
    return len(positives), median, auc