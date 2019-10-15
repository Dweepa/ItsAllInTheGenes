import tensorflow as tf
import pickle
import sys
import numpy as np
import pandas as pd

# d k embedding_length epoch epoch epoch...

n_layers = int(sys.argv[1])
n_units = int(sys.argv[2])
embedding_length = int(sys.argv[3])
epochs = [int(a) for a in sys.argv[4:]]

model_name = "MOD_snn_"+str(n_layers)+"_"+str(n_units)+"_"+str(embedding_length)
embedding_name =  "EMB_snn_"+str(n_layers)+"_"+str(n_units)+"_"+str(embedding_length)

def save_embeddings(X, y, model_name, epoch):
    embeddings = []
    with tf.Session() as session:
        saver = tf.train.import_meta_graph('../Models/'+model_name+'/'+model_name+'-'+str(epoch)+'.meta')
        saver.restore(session, '../Models/'+model_name+'/'+model_name+'-'+str(epoch))
        graph = tf.get_default_graph()
        original_input = graph.get_tensor_by_name('gene_expression:0')
        norm_embeddings = graph.get_tensor_by_name('norm_embeddings:0')
        print(original_input)
        print("Loaded")
        for a in range(len(X)):
            if a%1000==0:
                sys.stdout.write("\r%d/%d" % (a, len(X)))
            feed_dict={original_input:np.asarray(X[a:a+1])}
            curr_embedding = session.run([norm_embeddings], feed_dict=feed_dict)[0][0]
            embeddings.append(list(curr_embedding)+list([y[a]]))
        embeddings = pd.DataFrame(embeddings, columns=['e'+str(a) for a in range(1, embedding_length+1)]+['pert_id'])
        sys.stdout.write("\r%d/%d\nCompleted\n" % (len(X), len(X)))
        # pickle.dump(embeddings, open(filename, 'wb'))

print("Loaded Modules")
print("Loading Data")
data = pickle.load(open('../Data/full', 'rb'))

print(f"Data Loaded\nNumber of Columns: {len(data.columns)}\nNumber of Rows: {len(data)}")

X = data.loc[:, '780':'79716']
y = list(data['target'])

for a in epochs:
    save_embeddings(X, y, './embeddings/'+embeddingname)