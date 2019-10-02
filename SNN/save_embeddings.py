import tensorflow as tf
import pickle
import sys
import numpy as np
import pandas as pd

modelname = sys.argv[1]
embeddingname = sys.argv[2]

def save_embeddings(X, y, filename):
    with tf.Session() as session:
        saver = tf.train.import_meta_graph('./models/'+modelname+'/'+modelname+'.meta')
        saver.restore(session,tf.train.latest_checkpoint('./models/'+modelname))
        graph = tf.get_default_graph()
        original_input = graph.get_tensor_by_name('gene_expression:0')
        norm_embeddings = graph.get_tensor_by_name('norm_embeddings:0')
        print(original_input)
        print("Loaded")
        feed_dict={original_input:np.asarray(X)}
        embeddings = pd.DataFrame(session.run([norm_embeddings], feed_dict=feed_dict)[0],
                                  columns=['e'+str(a) for a in range(1, 33)])
        embeddings['target'] = y
        pickle.dump(embeddings, open(filename, 'wb'))

print("Loaded Modules")
print("Loading Data")
data = pickle.load(open('../Data/full', 'rb'))
print(f"Data Loaded\nNumber of Columns: {len(data.columns)}\nNumber of Rows: {len(data)}")

X = data.loc[:, '780':'79716'][:10000]
y = list(data['target'])[:10000]

save_embeddings(X, y, './embeddings/'+embeddingname)