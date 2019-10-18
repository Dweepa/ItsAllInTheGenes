import tensorflow as tf
import pickle
import sys
import numpy as np
import pandas as pd

# d k embedding_length epoch epoch epoch...
# TODO: make this work
# command-line arguments
layer = int(sys.argv[1])
neuron = int(sys.argv[2])
emb_len = int(sys.argv[3])
dropout = float(sys.argv[4])
samples_per_pert = int(sys.argv[5])
only_test = int(sys.argv[6])
epochs = [int(a) for a in sys.argv[7:]]

model_name = "MOD_triplet_" + str(layer) + "_" + str(neuron) + "_" + str(emb_len) + "_" + str(dropout) + "_" + str(
    samples_per_pert)
embedding_name = "EMB_triplet_" + str(layer) + "_" + str(neuron) + "_" + str(emb_len) + "_" + str(dropout) + "_" + str(
    samples_per_pert)

print(model_name, embedding_name, only_test)


embedding_length = emb_len


def save_embeddings(X, y, model_name, epoch):
    embeddings = []
    with tf.Session() as session:
        saver = tf.train.import_meta_graph('../Models/' + model_name + '/' + model_name + '-' + str(epoch) + '.meta')
        saver.restore(session, '../Models/' + model_name + '/' + model_name + '-' + str(epoch))
        graph = tf.get_default_graph()
        # print([n.name for n in tf.get_default_graph().as_graph_def().node])
        print(tf.all_variables())
        # original_input = graph.get_tensor_by_name('input:0')

        W = graph.get_tensor_by_name('siamese/fc_embeddingW:0')
        b = graph.get_tensor_by_name('siamese/fc_embeddingb:0')
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        norm_embeddings = graph.get_tensor_by_name('siamese/fc_embeddingW:0')

        # print("Loaded "+model_name+"-"+str(epoch))
        for a in range(len(X)):
            if a % 1000 == 0:
                sys.stdout.write("\r%d/%d" % (a, len(X)))
            feed_dict = {original_input: np.asarray(X[a:a + 1])}
            curr_embedding = session.run([norm_embeddings], feed_dict=feed_dict)[0][0]
            embeddings.append(list(curr_embedding) + list([y[a]]))
        embeddings = pd.DataFrame(embeddings,
                                  columns=['e' + str(a) for a in range(1, embedding_length + 1)] + ['pert_id'])
        embeddings.to_csv("../Embeddings/" + embedding_name + "-" + str(epoch))
        sys.stdout.write("\r%d/%d\nCompleted\n" % (len(X), len(X)))


# print("Loaded Modules")
# print("Loading Data")
# data = pickle.load(open('../Data/full', 'rb'))

# print(f"Data Loaded\nNumber of Columns: {len(data.columns)}\nNumber of Rows: {len(data)}")

if only_test == 1:
    X = pickle.load(open('../Data/SNN_triplet_X_test', 'rb'))
    y = pickle.load(open('../Data/SNN_triplet_y_test', 'rb'))

else:
    data = pickle.load(open('../Data/full', 'rb'))
    X = data.loc[:, '780':'79716']
    y = list(data['target'])

for epoch in epochs:
    save_embeddings(X, y, model_name, epoch)
