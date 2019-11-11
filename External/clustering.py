import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
import random
from sklearn.metrics import roc_curve, roc_auc_score, auc
import sklearn.metrics as metrics
import sys
from collections import Counter
from sklearn.decomposition import PCA
import pickle
from sklearn.cluster import KMeans

embedding_name = sys.argv[1]

embedding_length = int(embedding_name.split("_")[4].split("-")[0])
# print(embedding_name,embedding_length)
# data_array = normalize(np.random.rand(10000, embedding_length))
# data = pd.DataFrame(data_array,	columns = ['e'+str(a+1) for a in range(embedding_length)])

# data["pert_id"] = [chr(random.randint(0, 26)+65) for a in range(10000)]
# data["pert_class"] = [random.randint(0, 10) for a in range(10000)]

data = pd.read_csv(embedding_name)
# print(len(data))
external_data = pd.read_csv("./Final/all3_without_nan.csv")
perturbagens = [key for key, value in dict(Counter(external_data.id)).items() if value==1]

data = data[data.pert_id.isin(perturbagens)]

pert_class = {}
for ind in external_data.index:
	pert_class[external_data.id[ind]] = external_data.atc[ind][:1]

data["pert_class"] = [pert_class[data.pert_id[ind]] for ind in data.index]
# print(data["pert_class"])
# print(data.head())
# print(len(data))


# K-Means
X_full = data[['e1','e2','e3','e4','e5','e6','e7','e8','e9','e10','e11','e12','e13','e14','e15','e16']]
# print(Counter(data["pert_class"]), len(data['pert_class']))

# Getting perturbagen-level embeddings
for i in np.unique(data['pert_class']):
	X = data[data['pert_class']==i].mean()
	print("pert", X)
	break

# PCA
pca = PCA(n_components=2)
princi = pca.fit_transform(X)
# print(princi)

# Label encoding
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le = le.fit(data['pert_class']).transform(data['pert_class'])
# print(data["pert_class"],le)

# CLustering
kmeans = KMeans(14, random_state=0)
labels = kmeans.fit(princi,le)
# print(labels)
pred = labels.predict(princi)
# print("pred: ",Counter(pred))
# print("true: ",Counter(le))
# print("Accuracy: ",metrics.accuracy_score(pred,le))
plt.scatter(princi[:, 0], princi[:, 1], c=le, s=40, cmap='viridis')
plt.show()

# print(princi[:,0])
