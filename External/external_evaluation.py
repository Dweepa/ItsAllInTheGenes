import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
import random
from sklearn.metrics import roc_curve, roc_auc_score, auc
import sklearn.metrics as metrics
import sys
from collections import Counter

embedding_name = sys.argv[1]

embedding_length = int(embedding_name.split("/")[-1].split("_")[4].split("-")[0])
print(embedding_length)

# data_array = normalize(np.random.rand(10000, embedding_length))
# data = pd.DataFrame(data_array,	columns = ['e'+str(a+1) for a in range(embedding_length)])

# data["pert_id"] = [chr(random.randint(0, 26)+65) for a in range(10000)]
# data["pert_class"] = [random.randint(0, 10) for a in range(10000)]

data = pd.read_csv(embedding_name)

external_data = pd.read_csv("./Final/all3_without_nan.csv")
perturbagens = [key for key, value in dict(Counter(external_data.id)).items() if value==1]

data = data[data.pert_id.isin(perturbagens)]

pert_class = {}
for ind in external_data.index:
	pert_class[external_data.id[ind]] = external_data.atc[ind][:1]

data["pert_class"] = [pert_class[data.pert_id[ind]] for ind in data.index]
# print(data.head())
# print(len(data))

def softmax_function(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()


def label_positive(row, label, y):
	if y[int(row['index'])] == label:
		return True
	return False

def external_evaluate(data, embedding_length, sample_size, number_test_cases):
	imp_q = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
	all_keys = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, "median", "auc"]

	all_outputs = {}
	for key in all_keys:
		all_outputs[key] = []
	print("HELLo")
	for a in range(number_test_cases):
		output = [np.mean(all_outputs[key]) for key in all_keys]
		sys.stdout.write("\r%d/%d AUC: %f Top-0.1: %f" % (a, number_test_cases, output[-1], output[-5]))
		data_small = data.sample(sample_size)

		X = np.asarray(data_small.loc[:, 'e1':'e' + str(embedding_length)])
		y = np.asarray(data_small['pert_class'])
		ind = np.random.choice(len(y), 1)
		query_embedding = X[ind]
		query_class = y[ind]

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

		results = pd.DataFrame(list(zip(imp_q, imp_q_val)), columns=['Quantile', 'Recall'])
			
		auc = roc_auc_score(ordered['label'], ordered['softmax'])
		
		imp_q_val['median'] = median
		imp_q_val['auc'] = auc
		
		for key in all_keys:
			all_outputs[key].append(imp_q_val[key])


	sys.stdout.write("\r%d/%d\nCompleted\n" % (number_test_cases, number_test_cases))
	output = [np.mean(all_outputs[key]) for key in all_keys]
	return output

print(external_evaluate(data, embedding_length, 10000, 500))

