import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import random
import sys
from collections import Counter

embedding_name = sys.argv[1]
net_type = embedding_name.split("_")[1]

if net_type=='snn':
    d = int(embedding_name.split("_")[2])
    k = int(embedding_name.split("_")[3])
    embedding_length = int(embedding_name.split("_")[4].split('-')[0])
else:
    pass

embeddings = pd.read_csv('../Embeddings/' + embedding_name)

X = np.asarray(embeddings.loc[:, 'e1':'e' + str(embedding_length)])
y = np.asarray(embeddings['pert_id'])
perturbagens = np.unique(y)

imp_q = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
all_keys = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, "median", "auc"]


def softmax_function(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def label_positive(row, label, y):
    if y[int(row['index'])] == label:
        return True
    return False


def get_set(number, X, y):
    selection = perturbagens[np.random.choice(len(perturbagens), number, replace=False)]
    ind = [y[a] in selection for a in range(len(y))]
    return X[ind], y[ind]


def full_internal_evaluation(query_embedding, query_class, X, y, printinfo=False):
    cosines = np.dot(X, query_embedding.reshape(embedding_length, 1))
    softmax = softmax_function(cosines).flatten()
    sorted_softmax = list(zip(softmax, range(len(y))))
    sorted_softmax.sort(reverse=True)
    ordered = pd.DataFrame(sorted_softmax, columns=['softmax', 'index'])
    ordered['label'] = ordered.apply(lambda row: label_positive(row, query_class, y), axis=1)
    positives = ordered.index[ordered['label'] == True].tolist()
    quantiles = []
    pos = 0
    total_negs = len(y) - len(positives)
    for a in range(len(positives)):
        quantiles.append((positives[a] - pos) / total_negs)
        pos += 1

    x_quantile = {}
    for a in np.arange(0, 1.001, 0.001):
        x_quantile[round(a, 3)] = 0
    # y_recall = []
    for a in quantiles:
        x_quantile[round(a, 3)] += (1 / len(quantiles))
    for a in np.arange(0.001, 1.001, 0.001):
        x_quantile[round(a, 3)] += x_quantile[round(a - 0.001, 3)]
    x_values = x_quantile.keys()
    y_values = x_quantile.values()

    imp_q_val = dict()
    for a in imp_q:
        imp_q_val[a] = x_quantile[round(a, 3)]

    median = np.median(quantiles)
    if printinfo:
        print("Median Quantile:", median)

    results = pd.DataFrame(list(zip(imp_q, imp_q_val)), columns=['Quantile', 'Recall'])
    
    if printinfo:
        display(HTML(results.to_html()))
        plt.figure(figsize=[20, 6])
        plt.xlabel("Quantile")
        plt.ylabel("Percentage of positive perturbagens below quantile")
        plt.plot(x_values, y_values)
        plt.show()
    
    auc = roc_auc_score(ordered['label'], ordered['softmax'])
    
    if printinfo:
        #		 plt.figure(figsize=[20,6])
        #		 fpr, tpr, thresholds = roc_curve(ordered['label'], ordered['softmax'])
        #		 plt.plot(fpr, tpr)
        #		 plt.show()

        plt.figure(figsize=[20, 6])
        l = 100  # positives[-10]
        colors = {True: 'green', False: 'red'}
        plt.scatter([a for a in range(l)], [0 for a in range(l)], c=ordered['label'][:l].apply(lambda x: colors[x]))
        plt.show()

        plt.figure(figsize=[20, 6])
        l = 100  # positives[-10]
        plt.scatter([a for a in range(l)], [0 for a in range(l)], c=[a[1] for a in sorted_softmax[:l]])
        plt.show()

        print("AUC: ", auc)
    imp_q_val['median'] = median
    imp_q_val['auc'] = auc
    return imp_q_val


all_outputs = dict()
for a in imp_q:
    all_outputs[a] = []
all_outputs['median'] = []
all_outputs['auc'] = []

test_cases = 10
for a in range(test_cases):
    sys.stdout.write("\r%d/%d" % (a, test_cases))
    X_small, y_small = get_set(100, X, y)
    ind = np.random.choice(len(y_small), 1)
    imp_q_val = full_internal_evaluation(X_small[ind], y_small[ind], X_small, y_small, printinfo=False)
    for key in imp_q_val.keys():
        all_outputs[key].append(imp_q_val[key])
sys.stdout.write("\r%d/%d\n" % (a + 1, test_cases))


if net_type=="snn":
    file = open("../Results/SNN_results")
    parameter_count = str(978*d*k + 0.5*k*k*d*(d-1))

    outputs = [parameter_count, str(d), str(k), str(embedding_length), 'MOD'+embedding_name[3:], embedding_name]

else:
    file = open("../Results/Triplet_results")

for a in all_keys:
    outputs.append(str(np.mean(all_outputs[a])))

file.write(",".join(outputs))
file.close()
