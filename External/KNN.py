import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import random
from sklearn.metrics import roc_curve, roc_auc_score, auc
import sklearn.metrics as metrics
import sys
from collections import Counter

embedding_name = sys.argv[1]

embedding_length = int(embedding_name.split("_")[4].split("-")[0])

data = pd.read_csv(embedding_name)

external_data = pd.read_csv("./Final/all3_without_nan.csv")
perturbagens = [key for key, value in dict(Counter(external_data.id)).items() if value==1]

data = data[data.pert_id.isin(perturbagens)]

pert_class = {}
for ind in external_data.index:
	pert_class[external_data.id[ind]] = external_data.atc[ind][:1]

data["pert_class"] = [pert_class[data.pert_id[ind]] for ind in data.index]

all_embeddings = data[['e'+str(a) for a in range(1, embedding_length+1)]]

perts = np.unique(data["pert_id"])
pert_embeddings = {}
pert_classes = {}


for pert in perts:
	embedding_coll = np.asarray(all_embeddings[data["pert_id"]==pert])
	pert_embeddings[pert] = embedding_coll.mean(axis=0)
	pert_class[pert] = data[pert]


def find(test_pert, pert_embeddings, pert_class, num):
	similarity = []
	test_embedding = pert_embeddings[test_pert]

	for pert in pert_embeddings.keys():
		embedding = pert_embeddings[pert]
		similarity.append((np.dot(test_embedding, embedding), pert))

	similarity.sort(reverse=True)

	similar_perts = [pert_class[pert] for sim, pert in similarity[:num]]

	print(Counter(similar_perts))



find(perts[16], pert_embeddings, pert_class, 100)
