import sys
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.datasets import mnist
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import math
from sklearn.model_selection import train_test_split, KFold
import os
import random
import subprocess

# #importing dataset
# print("Loaded Modules")
# print("Loading Data")
# data = pickle.load(open('../Data/full', 'rb'))
# print(f"Data Loaded\nNumber of Columns: {len(data.columns)}\nNumber of Rows: {len(data)}")

# X = data.loc[:, '780':'79716']
# y = list(data['target'])
# pert_dict = {}
# num = 0
# for a in range(len(y)):
#	 if y[a] not in pert_dict.keys():
#		 pert_dict[y[a]] = num
#		 num+=1
#	 y[a] = pert_dict[y[a]]
# all_pert = np.unique(y)

# #end of loading dataset

all_pert = np.asarray([a for a in range(65, 85)])
X = pd.DataFrame(np.random.rand(2000, 978))
y = np.asarray([random.choice(all_pert) for a in range(2000)])


# Making the graph

input_size = 978
n_layers = 16
n_classes = 2170
n_units = 21
batch_size = 3000
epochs = 20
learning_rate = 0.005

embedding_length = 32
number_of_samples = 300
saving_multiple = 5

print("Creating tensorflow graph")
tf.reset_default_graph()

inputs = tf.placeholder(tf.float32, [None, input_size], name='gene_expression')
original_input = inputs
labels = tf.placeholder(tf.int32, [None], name='labels')

for a in range(n_layers):
	layer = tf.layers.dense(inputs, n_units, 'selu', name='layer'+str(a))
	inputs = tf.concat([inputs, layer], 1, name='concatenation'+str(a))

embeddings = tf.layers.dense(inputs, embedding_length, None, name='embedding')
norm_embeddings = tf.nn.l2_normalize(embeddings, axis=1, name='norm_embeddings')

# class_weights = tf.constant(normalize(np.random.rand(embedding_length, n_classes)), dtype=tf.float32,
#							name='class_weights')

class_weights = tf.Variable(tf.random_normal([embedding_length, n_classes], stddev=0.35), name="class_weights")

margin = tf.placeholder(tf.float32, name='margin')

alpha_initial = tf.get_variable(dtype=tf.float32, initializer=tf.constant(np.random.rand(1), dtype='float32'), 
								name='alpha')
alpha = tf.nn.relu(alpha_initial)

cosines = tf.matmul(norm_embeddings, class_weights, name='cosines')
onehot_label = tf.one_hot(labels, n_classes, name='labels_onehot')
m_onehot = tf.math.multiply(margin, onehot_label, name='mxonehot')
margin_cosine = tf.subtract(cosines, m_onehot, name='cosine-m')
alpha_margin_cosine = tf.math.exp(tf.math.multiply(alpha, margin_cosine), name='alphaxc-m')
amc_numerator = tf.reduce_sum(tf.multiply(alpha_margin_cosine, onehot_label), axis=1, name='amc_positive')
amc_denominator = tf.reduce_sum(alpha_margin_cosine, axis=1, name='amc_total')
amc_fraction = tf.divide(amc_numerator, amc_denominator, name='amc_fraction')
log_amcf = -tf.log(amc_fraction, name='log_amc')
loss = tf.reduce_sum(log_amcf, name='loss')

m = 0
m_change = 0.00002
m_max = 25

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

print("Graph generated")
saver = tf.train.Saver(max_to_keep=2)

#End of making the Graph

def run_network(session, epochs, X_train, X_val, y_train, y_val):
	feed_dict={original_input:np.asarray(X_train), labels: np.asarray(y_train), margin: m}	
	total = np.asarray(session.run([original_input], feed_dict=feed_dict)).shape[1]
	order = np.arange(total)
	losses = []
	for a in range(epochs):
		np.random.shuffle(order)
		total_loss = 0
		sys.stdout.write("\rEpoch %d: \t[%s%s] %d/%d Loss: %f" % (a, "="*0, ' '*(50), 0, total, 0))
			
		for ind in range(0, total, batch_size):
			sys.stdout.flush()
			perc = math.ceil(50*(ind/total))
			feed_dict[original_input] = X_train.iloc[order[ind:ind+batch_size]]
			feed_dict[labels] = [y_train[a] for a in order[ind:ind+batch_size]]
			_, l = session.run([optimizer, loss], feed_dict=feed_dict)
			sys.stdout.write("\rEpoch %d: \t[%s%s] %d/%d Loss: %f" % (a, "="*perc, ' '*(50-perc), ind, total, total_loss/(ind+batch_size)))
			total_loss+=(l)
			
		sys.stdout.write("\rEpoch %d: \t[%s] %d/%d Loss: %f\n" % (a, "="*50, total, total, total_loss/total))
		losses.append(total_loss/total)
		feed_dict[margin] = min(feed_dict[margin]+m_change, m_max)

	embeddings = []
	for a in range(len(X_val)):
		if a%10==0:
			sys.stdout.write("\r%d/%d" % (a, len(X_val)))
		feed_dict={original_input:np.asarray(X_val[a:a+1])}
		curr_embedding = session.run([norm_embeddings], feed_dict=feed_dict)[0][0]
		embeddings.append(list(curr_embedding)+list([y_val[a]]))
	embeddings = pd.DataFrame(embeddings, columns=['e'+str(a) for a in range(1, 33)]+['target'])
	pickle.dump(embeddings, open('./embeddings/cross_validation_emb', 'wb'))

	print("Testing Embeddings")
	subprocess.run(["python", "internal_evaluation.py", "cross_validation_emb"])

	print(embeddings.shape)

def cross_validate(session, splitsize):
	kf = KFold(n_splits=splitsize)
	results = []
	a = 1
	for train_idx, val_idx in kf.split(all_pert):
		print("Validation "+str(a))
		a+=1
		train_perts = all_pert[train_idx]
		val_perts = all_pert[val_idx]

		train_indices = [a in train_perts for a in y]
		val_indices = [a in val_perts for a in y]

		X_train = X[train_indices]
		X_val = X[val_indices]
		y_train = y[train_indices] 
		y_val = y[val_indices]

		run_network(session, epochs, X_train, X_val, y_train, y_val)

with tf.Session() as session:
	alpha_initial.initializer.run()
	tf.initialize_all_variables().run()
	print("Initialized")

	result = cross_validate(session,5)

	# print ("Cross-validation result: %s" % result)
