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
from sklearn.model_selection import train_test_split
import os
from data import *

# parameters modelname d k embeddingname embedding_length
# python3 SNN.py 

# -- Dweepa Code ----
full = pickle.load(open('../Data/full', 'rb'))
test_pert = pickle.load(open('../Data/test_perts', 'rb'))
train_pert = pickle.load(open('../Data/train_perts', 'rb'))

# List of all perturbagens
all_pert = np.concatenate((train_pert, test_pert))

# Get train and test perturbagen
train_pert, test_pert = train_and_test_perturbagens(all_pert, 95)

# Generate Data
X_train, y_train, X_test, y_test = generate_data(full, train_pert, test_pert)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# --- End of Dweepa Code ----


n_layers = int(sys.argv[1])
n_units = int(sys.argv[2])
embedding_length = int(sys.argv[3])
model_name = "MOD_snn_"+str(n_layers)+"_"+str(n_units)+"_"+str(embedding_length)

os.mkdir('../Models/'+model_name)

# print("Loaded Modules")
# print("Loading Data")
data = pickle.load(open('../Data/full', 'rb'))
# print(f"Data Loaded\nNumber of Columns: {len(data.columns)}\nNumber of Rows: {len(data)}")

X = data.loc[:, '780':'79716']
y = list(data['target'])
pert_dict = {}
num = 0
for a in range(len(y)):
    if y[a] not in pert_dict.keys():
        pert_dict[y[a]] = num
        num+=1
    y[a] = pert_dict[y[a]]

X = X[:100]
y = np.asarray(y[:100]).flatten()
# y_small

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

input_size = 978
# n_layers = 16
n_classes = 2170
# n_units = 21
batch_size = 3000
epochs = 100
learning_rate = 0.005

# embedding_length = 32
number_of_samples = 300
saving_multiple = 25

# print("Creating tensorflow graph")
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
#                            name='class_weights')

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

# print("Graph generated")
saver = tf.train.Saver(max_to_keep=3)
with tf.Session() as session:    
    feed_dict={original_input:X, labels: y, margin: m}    
    
    alpha_initial.initializer.run()
    tf.initialize_all_variables().run()
    # print("Initialized")
    
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
            feed_dict[original_input] = X.iloc[order[ind:ind+batch_size]]
            feed_dict[labels] = [y[a] for a in order[ind:ind+batch_size]]
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            sys.stdout.write("\rEpoch %d: \t[%s%s] %d/%d Loss: %f" % (a, "="*perc, ' '*(50-perc), ind, total, total_loss/(ind+batch_size)))
            total_loss+=(l)
            
        sys.stdout.write("\rEpoch %d: \t[%s] %d/%d Loss: %f\n" % (a, "="*50, total, total, total_loss/total))
        losses.append(total_loss/total)
        feed_dict[margin] = min(feed_dict[margin]+m_change, m_max)
    
    # tf.summary.FileWriter('./logs', session.graph)
        if a%saving_multiple==0:
            saver.save(session, '../Models/'+model_name+"/"+model_name, global_step=a)
    saver.save(session, '../Models/'+model_name+"/"+model_name, global_step=epochs)

