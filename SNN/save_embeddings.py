import tensorflow as tf
import pickle
import sys
import numpy as np
import pandas as pd

modelname = sys.argv[1]
embeddingname = sys.argv[2]

def save_embeddings(X, y, filename):
    embeddings = []
    with tf.Session() as session:
        saver = tf.train.import_meta_graph('./models/'+modelname+'/'+modelname+'.meta')
        saver.restore(session,tf.train.latest_checkpoint('./models/'+modelname))
        graph = tf.get_default_graph()
        original_input = graph.get_tensor_by_name('gene_expression:0')
        norm_embeddings = graph.get_tensor_by_name('norm_embeddings:0')
        print(original_input)
        print("Loaded")
        for a in range(len(X)):
            sys.stdout.write("\r%d/%d" % (a, len(X)))
            feed_dict={original_input:np.asarray(X[a:a+1])}
            curr_embedding = session.run([norm_embeddings], feed_dict=feed_dict)[0][0]
            embeddings.append(list(curr_embedding)+list([y[a]]))
        embeddings = pd.DataFrame(embeddings, columns=['e'+str(a) for a in range(1, 33)]+['target'])
        sys.stdout.write("\r%d/%d\nCompleted\n" % (a, len(X)))
        pickle.dump(embeddings, open(filename, 'wb'))

print("Loaded Modules")
print("Loading Data")
data = pickle.load(open('../Data/full', 'rb'))
print(f"Data Loaded\nNumber of Columns: {len(data.columns)}\nNumber of Rows: {len(data)}")

X = data.loc[:, '780':'79716'][:10]
y = list(data['target'])[:10]

save_embeddings(X, y, './embeddings/'+embeddingname)