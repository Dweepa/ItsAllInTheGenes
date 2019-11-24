import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import random
from sklearn.metrics import roc_curve, roc_auc_score, auc
import sklearn.metrics as metrics
import sys
from collections import Counter
from scipy.spatial.distance import cosine


embedding_name = sys.argv[1]

embedding_length = int(embedding_name.split("/")[-1].split("_")[4].split("-")[0])

data = pd.read_csv(embedding_name)

external_data = pd.read_csv("./Final/all3_without_nan.csv")
perturbagens = [key for key, value in dict(Counter(external_data.id)).items() if value==1]

data = data[data.pert_id.isin(perturbagens)]

pert_class = {}
for ind in external_data.index:
	pert_class[external_data.id[ind]] = external_data.atc[ind][:1]

data["pert_class"] = [pert_class[data.pert_id[ind]] for ind in data.index]

data = data[data["pert_class"].isin(["N", "C", "L", "A"])]

all_embeddings = data[['e'+str(a) for a in range(1, embedding_length+1)]]

perts = np.unique(data["pert_id"])
pert_embeddings = {}

for pert in perts:
	embedding_coll = np.asarray(all_embeddings[data["pert_id"]==pert])
	pert_embeddings[pert] = embedding_coll.mean(axis=0)

distribution = dict(Counter(data["pert_class"]))
for key in distribution.keys():
	distribution[key] = 1/distribution[key]


def find(test_pert, pert_embeddings, pert_class, num, weighted=False, use_all=False, distribution=None):
	similarity = []
	test_embedding = pert_embeddings[test_pert]

	for pert in pert_embeddings.keys():
		if pert==test_pert:
			continue
		embedding = pert_embeddings[pert]
		similarity.append((cosine(test_embedding, embedding), pert))

	similarity.sort(reverse=False)

	if weighted==False:
		if use_all==False:
			similar_perts = [pert_class[pert] for sim, pert in similarity[:num]]
			new_similarity = Counter(similar_perts)
			if distribution!=None:
				for a in new_similarity.keys():
					new_similarity[a] = new_similarity[a]*distribution[a]
			# print(new_similarity)
			return new_similarity.most_common()[0][0]
		else:
			answer = []
			for a in range(1, num+1):
				similar_perts = [pert_class[pert] for sim, pert in similarity[:a]]
				answer.append(Counter(similar_perts).most_common()[0][0])
			return answer
	
	else:
		if use_all==True:
			answer = []
			for ind in range(1, num+1):
				similar_perts = [(pert_class[pert], sim) for sim, pert in similarity[:ind]]
				new_similarity = {}
				for class_val, sim in similar_perts:
					if class_val not in new_similarity.keys():
						new_similarity[class_val] = 0
					new_similarity[class_val]+=sim
				maxval = -1
				maxkey = -1
				for key in new_similarity.keys():
					if maxval==-1 or new_similarity[key]>maxval:
						maxkey = key
						maxval = new_similarity[key]
				answer.append(maxkey)
			return answer
		else:
			similar_perts = [(pert_class[pert], sim) for sim, pert in similarity[:num]]
			new_similarity = {}
			for class_val, sim in similar_perts:
				if class_val not in new_similarity.keys():
					new_similarity[class_val] = 0
				new_similarity[class_val]+=sim
			maxval = -1
			maxkey = -1
			for key in new_similarity.keys():
				if maxval==-1 or new_similarity[key]>maxval:
					maxkey = key
					maxval = new_similarity[key]
			# print(similar_perts, new_similarity)
			return maxkey



totry = 10
accuracy = [0 for _ in range(totry)]

for num in range(0, len(perts)):
	guesses = find(perts[num], pert_embeddings, pert_class, totry, weighted=False, use_all=True)
	answer = pert_class[perts[num]]

	for a in range(totry):
		if guesses[a]==answer:
			accuracy[a]+=1

for a in range(totry):
	print(a+1, accuracy[a]/len(perts))

# total = 0
# for num in range(0, len(perts)):
# 	guess = find(perts[num], pert_embeddings, pert_class, 10, weighted=False, use_all=False)
# 	answer = pert_class[perts[num]]
# 	if guess==answer:
# 		total+=1
# print(total/len(perts))

# blah = random.randint(0, len(perts))
# print(blah, pert_class[perts[blah]])
# guess = find(perts[blah], pert_embeddings, pert_class, 10, weighted=False, use_all=False)
# print(guess)
# guess = find(perts[blah], pert_embeddings, pert_class, 10, weighted=True, use_all=False)
# print(guess)